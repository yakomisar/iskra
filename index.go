// Package index implements an inverted index for full-text search
//
// ═══════════════════════════════════════════════════════════════════════════════
// WHAT IS AN INVERTED INDEX?
// ═══════════════════════════════════════════════════════════════════════════════
// An inverted index is like the index at the back of a book, but for search engines.
//
// Example: Given these documents:
//
//	Doc 1: "the quick brown fox"
//	Doc 2: "the lazy dog"
//	Doc 3: "quick brown dogs"
//
// The inverted index would look like:
//
//	"quick"  → [Doc1:Pos1, Doc3:Pos0]
//	"brown"  → [Doc1:Pos2, Doc3:Pos1]
//	"fox"    → [Doc1:Pos3]
//	"lazy"   → [Doc2:Pos1]
//	"dog"    → [Doc2:Pos2]
//	"dogs"   → [Doc3:Pos2]
//
// ═══════════════════════════════════════════════════════════════════════════════
package iskra

import (
	"errors"
	"log/slog"
	"sync"

	"github.com/RoaringBitmap/roaring"
)

// ═══════════════════════════════════════════════════════════════════════════════
// ERROR DEFINITIONS
// ═══════════════════════════════════════════════════════════════════════════════
var (
	ErrNoPostingList = errors.New("no posting list exists for token")
	ErrNoNextElement = errors.New("no next element found")
	ErrNoPrevElement = errors.New("no previous element found")
)

// ═══════════════════════════════════════════════════════════════════════════════
// BM25 RANKING SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

// BM25Parameters holds the tuning parameters for BM25 algorithm
type BM25Parameters struct {
	K1 float64 // Term frequency saturation (typical: 1.2-2.0)
	B  float64 // Length normalization (typical: 0.75)
}

// DefaultBM25Parameters returns the standard BM25 parameters
func DefaultBM25Parameters() BM25Parameters {
	return BM25Parameters{
		K1: 1.5,
		B:  0.75,
	}
}

// DocumentStats stores statistics about a single document.
// DocID is uint32 to match the index-wide convention.
type DocumentStats struct {
	DocID     uint32         // Document identifier
	Length    int            // Number of terms in the document
	TermFreqs map[string]int // How many times each term appears
}

// ═══════════════════════════════════════════════════════════════════════════════
// INDEX OPTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// IndexOption is a functional option for Index / Upsert calls.
type IndexOption func(*indexOptions)

type indexOptions struct {
	disableAnalyzer bool
}

// WithoutAnalyzer disables text analysis for a single Index/Upsert call.
// The document is stored as a single token exactly as provided — no lowercasing,
// no stop-word removal, no stemming.
//
// Useful when documents are already pre-processed, contain structured identifiers,
// or when you need exact-match behaviour without any normalization.
//
// Example:
//
//	idx.Index(42, "SKU-9021-XL", iskra.WithoutAnalyzer())
//	idx.Upsert(42, "SKU-9021-XL", iskra.WithoutAnalyzer())
func WithoutAnalyzer() IndexOption {
	return func(o *indexOptions) {
		o.disableAnalyzer = true
	}
}

func applyIndexOptions(opts []IndexOption) indexOptions {
	o := indexOptions{}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORE DATA STRUCTURE: InvertedIndex with HYBRID STORAGE
// ═══════════════════════════════════════════════════════════════════════════════

// InvertedIndex is the main search index.
// DocStats and all DocID references use uint32.
type InvertedIndex struct {
	mu sync.Mutex

	// DOCUMENT-LEVEL STORAGE (for fast document lookups and boolean queries)
	DocBitmaps map[string]*roaring.Bitmap // Term → Bitmap of document IDs

	// POSITION-LEVEL STORAGE (for phrase search, proximity)
	PostingsList map[string]SkipList // Term → Positions

	// BM25 data
	DocStats   map[uint32]DocumentStats // DocID → statistics
	TotalDocs  int
	TotalTerms int64
	BM25Params BM25Parameters
}

// NewInvertedIndex creates a new empty inverted index
func NewInvertedIndex() *InvertedIndex {
	return &InvertedIndex{
		DocBitmaps:   make(map[string]*roaring.Bitmap),
		PostingsList: make(map[string]SkipList),
		DocStats:     make(map[uint32]DocumentStats),
		TotalDocs:    0,
		TotalTerms:   0,
		BM25Params:   DefaultBM25Parameters(),
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// INDEXING
// ═══════════════════════════════════════════════════════════════════════════════

// Index adds a document to the inverted index.
//
// By default the full analysis pipeline is applied (tokenization, lowercasing,
// stop-word removal, stemming). Pass WithoutAnalyzer() to skip all analysis and
// store the document as a single raw token.
//
// Example (default):
//
//	idx.Index(1, "The Quick Brown Fox")
//
// Example (raw, no analysis):
//
//	idx.Index(1, "SKU-9021-XL", iskra.WithoutAnalyzer())
func (idx *InvertedIndex) Index(docID uint32, document string, opts ...IndexOption) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	slog.Info("indexing document", slog.Uint64("docID", uint64(docID)))
	idx.indexDocumentUnlocked(docID, document, applyIndexOptions(opts))
}

// indexToken adds a single token occurrence to the index (HYBRID STORAGE)
func (idx *InvertedIndex) indexToken(token string, docID uint32, position int) {
	// Update roaring bitmap (document-level)
	if idx.DocBitmaps[token] == nil {
		idx.DocBitmaps[token] = roaring.NewBitmap()
	}
	idx.DocBitmaps[token].Add(docID)

	// Update skip list (position-level)
	skipList, exists := idx.getPostingList(token)
	if !exists {
		skipList = *NewSkipList()
	}
	skipList.Insert(Position{
		DocumentID: docID,
		Offset:     position,
	})
	idx.PostingsList[token] = skipList
}

// getPostingList retrieves the posting list for a token
func (idx *InvertedIndex) getPostingList(token string) (SkipList, bool) {
	skipList, exists := idx.PostingsList[token]
	return skipList, exists
}

// ═══════════════════════════════════════════════════════════════════════════════
// BASIC SEARCH OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// First returns the first occurrence of a token in the index
func (idx *InvertedIndex) First(token string) (Position, error) {
	skipList, exists := idx.getPostingList(token)
	if !exists {
		return EOFDocument, ErrNoPostingList
	}
	return skipList.Head.Tower[0].Key, nil
}

// Last returns the last occurrence of a token in the index
func (idx *InvertedIndex) Last(token string) (Position, error) {
	skipList, exists := idx.getPostingList(token)
	if !exists {
		return EOFDocument, ErrNoPostingList
	}
	return skipList.Last(), nil
}

// Next finds the next occurrence of a token after the given position
func (idx *InvertedIndex) Next(token string, currentPos Position) (Position, error) {
	if currentPos.IsBeginning() {
		return idx.First(token)
	}
	if currentPos.IsEnd() {
		return EOFDocument, nil
	}

	skipList, exists := idx.getPostingList(token)
	if !exists {
		return EOFDocument, ErrNoPostingList
	}

	nextPos, _ := skipList.FindGreaterThan(currentPos)
	return nextPos, nil
}

// Previous finds the previous occurrence of a token before the given position
func (idx *InvertedIndex) Previous(token string, currentPos Position) (Position, error) {
	if currentPos.IsEnd() {
		return idx.Last(token)
	}
	if currentPos.IsBeginning() {
		return BOFDocument, nil
	}

	skipList, exists := idx.getPostingList(token)
	if !exists {
		return BOFDocument, ErrNoPostingList
	}

	prevPos, _ := skipList.FindLessThan(currentPos)
	return prevPos, nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPSERT
// ═══════════════════════════════════════════════════════════════════════════════

// Upsert inserts a new document or replaces the existing one for the given docID.
//
// Accepts the same options as Index — in particular WithoutAnalyzer() to bypass
// the analysis pipeline and store the document text verbatim.
//
// Example (default):
//
//	idx.Upsert(42, "updated document text")
//
// Example (raw):
//
//	idx.Upsert(42, "SKU-9021-XL", iskra.WithoutAnalyzer())
func (idx *InvertedIndex) Upsert(docID uint32, document string, opts ...IndexOption) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	idx.removeDocumentUnlocked(docID)
	idx.indexDocumentUnlocked(docID, document, applyIndexOptions(opts))
}

// Delete removes a document from the index
//
// All positional entries, bitmaps bits, BM25 statistics, and DocStats for
// the given docID are cleaned up automatically under the index lock.
// Terms that have no remaining postings after the removal are pruned
// from both PostingsList and DocBitmaps, keeping the index compact.
//
// Returns true if the document existed and was removed, false if it was not
// present in the index.
//
// Example:
//
// removed := idx.Delete(42)
//
//	if !removed {
//	  log.Println("document 42 was not in the index")
//	}
func (idx *InvertedIndex) Delete(docID uint32) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	slog.Info("deleting document", slog.Uint64("docID", uint64(docID)))
	return idx.removeDocumentUnlocked(docID)
}

// removeDocumentUnlocked removes all indexed data for a document.
// Caller must hold idx.mu.
func (idx *InvertedIndex) removeDocumentUnlocked(docID uint32) bool {
	stats, ok := idx.DocStats[docID]
	if !ok {
		return false
	}

	for term := range stats.TermFreqs {
		idx.removeDocumentPostingsForTerm(term, docID)
	}

	idx.TotalDocs--
	idx.TotalTerms -= int64(stats.Length)
	delete(idx.DocStats, docID)
	return true
}

// indexDocumentUnlocked tokenizes (or not) and indexes the document.
// Caller must hold idx.mu.
func (idx *InvertedIndex) indexDocumentUnlocked(docID uint32, document string, o indexOptions) {
	var tokens []string
	if o.disableAnalyzer {
		// Store the raw document as a single token so it can still be looked up.
		// A single-element slice preserves the position-based machinery unchanged.
		tokens = []string{document}
	} else {
		tokens = Analyze(document)
	}

	docStats := DocumentStats{
		DocID:     docID,
		Length:    len(tokens),
		TermFreqs: make(map[string]int),
	}

	for position, token := range tokens {
		idx.indexToken(token, docID, position)
		docStats.TermFreqs[token]++
	}

	idx.DocStats[docID] = docStats
	idx.TotalDocs++
	idx.TotalTerms += int64(len(tokens))
}

// removeDocumentPostingsForTerm removes all positional entries of a term for a doc.
func (idx *InvertedIndex) removeDocumentPostingsForTerm(term string, docID uint32) {
	skipList, exists := idx.PostingsList[term]
	if !exists {
		return
	}

	var toRemove []Position
	cur := skipList.Head
	for cur != nil && cur.Tower[0] != nil {
		cur = cur.Tower[0]
		if cur.Key.DocumentID == docID {
			toRemove = append(toRemove, cur.Key)
		} else if cur.Key.DocumentID > docID {
			break
		}
	}

	for _, p := range toRemove {
		skipList.Delete(p)
	}

	if skipList.Head == nil || skipList.Head.Tower[0] == nil {
		delete(idx.PostingsList, term)
		delete(idx.DocBitmaps, term)
		return
	}

	idx.PostingsList[term] = skipList

	if bm := idx.DocBitmaps[term]; bm != nil {
		bm.Remove(docID)
		if bm.IsEmpty() {
			delete(idx.DocBitmaps, term)
		}
	}
}
