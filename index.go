// Package index implements an inverted index for full-text search
//
// ═══════════════════════════════════════════════════════════════════════════════
// WHAT IS AN INVERTED INDEX?
// ═══════════════════════════════════════════════════════════════════════════════
// An inverted index is like the index at the back of a book, but for search engines.
//
// Example: Given these documents:
//   Doc 1: "the quick brown fox"
//   Doc 2: "the lazy dog"
//   Doc 3: "quick brown dogs"
//
// The inverted index would look like:
//   "quick"  → [Doc1:Pos1, Doc3:Pos0]
//   "brown"  → [Doc1:Pos2, Doc3:Pos1]
//   "fox"    → [Doc1:Pos3]
//   "lazy"   → [Doc2:Pos1]
//   "dog"    → [Doc2:Pos2]
//   "dogs"   → [Doc3:Pos2]
//
// This allows us to:
// 1. Find documents containing a word instantly (without scanning all docs)
// 2. Find phrases by checking if word positions are consecutive
// 3. Rank results by how close words appear to each other (proximity)
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
// We define errors as package-level variables so they can be compared with ==
// This is a Go best practice for error handling.
var (
	ErrNoPostingList = errors.New("no posting list exists for token")
	ErrNoNextElement = errors.New("no next element found")
	ErrNoPrevElement = errors.New("no previous element found")
)

// ═══════════════════════════════════════════════════════════════════════════════
// BM25 RANKING SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════
// BM25 (Best Matching 25) is a ranking function used by search engines to estimate
// the relevance of documents to a given search query.
//
// WHY BM25?
// ---------
// 1. Industry standard: Used by Elasticsearch, Solr, Lucene
// 2. Accounts for document length (longer docs don't unfairly rank higher)
// 3. Accounts for term frequency saturation (10 vs 100 occurrences matter less)
// 4. Accounts for term rarity (rare terms are more significant)
//
// BM25 FORMULA:
// -------------
// For each term in the query:
//   score += IDF(term) * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (docLen / avgDocLen)))
//
// Where:
//   IDF = Inverse Document Frequency (how rare is this term?)
//   TF = Term Frequency (how often does term appear in this doc?)
//   k1 = Term frequency saturation parameter (typically 1.2-2.0)
//   b = Length normalization parameter (typically 0.75)
//   docLen = Length of this document
//   avgDocLen = Average document length in the corpus
//
// EXAMPLE:
// --------
// Query: "machine learning"
// Doc A: 100 words, contains "machine" 3 times, "learning" 2 times
// Doc B: 500 words, contains "machine" 5 times, "learning" 8 times
//
// Despite Doc B having more occurrences, Doc A might score higher because:
// 1. Doc A is shorter (length normalization)
// 2. The density of query terms is higher in Doc A
// ═══════════════════════════════════════════════════════════════════════════════

// BM25Parameters holds the tuning parameters for BM25 algorithm
type BM25Parameters struct {
	K1 float64 // Term frequency saturation (typical: 1.2-2.0)
	B  float64 // Length normalization (typical: 0.75)
}

// DefaultBM25Parameters returns the standard BM25 parameters
func DefaultBM25Parameters() BM25Parameters {
	return BM25Parameters{
		K1: 1.5,  // Moderate term frequency saturation
		B:  0.75, // Standard length normalization
	}
}

// DocumentStats stores statistics about a single document
type DocumentStats struct {
	DocID     int            // Document identifier
	Length    int            // Number of terms in the document
	TermFreqs map[string]int // How many times each term appears
}

// ═══════════════════════════════════════════════════════════════════════════════
// CORE DATA STRUCTURE: InvertedIndex with HYBRID STORAGE
// ═══════════════════════════════════════════════════════════════════════════════
// The InvertedIndex uses a hybrid approach for maximum efficiency:
//
// Architecture:
//
//	InvertedIndex
//	├── DocBitmaps: map[string]*roaring.Bitmap  (DOCUMENT-LEVEL)
//	│   ├── "quick" → Bitmap of document IDs [1, 3, 5, ...]
//	│   ├── "brown" → Bitmap of document IDs [1, 2, 7, ...]
//	│   └── "fox"   → Bitmap of document IDs [3, 5, ...]
//	├── PostingsList: map[string]SkipList       (POSITION-LEVEL)
//	│   ├── "quick" → SkipList of exact positions
//	│   ├── "brown" → SkipList of exact positions
//	│   └── "fox"   → SkipList of exact positions
//	└── mu: mutex for thread safety
//
// Why Hybrid Storage?
//   - Roaring Bitmaps: Lightning-fast for document-level operations (AND, OR, NOT)
//     10-100x memory compression, O(1) boolean operations
//   - Skip Lists: Essential for position-based queries (phrases, proximity)
//
// This gives us the best of both worlds!
// ═══════════════════════════════════════════════════════════════════════════════
type InvertedIndex struct {
	mu sync.Mutex // Protects against concurrent access

	// DOCUMENT-LEVEL STORAGE (for fast document lookups and boolean queries)
	DocBitmaps map[string]*roaring.Bitmap // Term → Bitmap of document IDs

	// POSITION-LEVEL STORAGE (for phrase search, proximity)
	PostingsList map[string]SkipList // Term → Positions

	// ===============================
	// BM25 INDEXING DATA STRUCTURES
	// ===============================
	DocStats   map[int]DocumentStats // DocID → statistics
	TotalDocs  int                   // Total number of indexed documents
	TotalTerms int64                 // Total number of terms across all docs
	BM25Params BM25Parameters        // BM25 tuning parameters
}

// NewInvertedIndex creates a new empty inverted index with hybrid storage and BM25 support
func NewInvertedIndex() *InvertedIndex {
	return &InvertedIndex{
		DocBitmaps:   make(map[string]*roaring.Bitmap), // Initialize document-level bitmaps
		PostingsList: make(map[string]SkipList),        // Initialize position-level skip lists
		DocStats:     make(map[int]DocumentStats),
		TotalDocs:    0,
		TotalTerms:   0,
		BM25Params:   DefaultBM25Parameters(),
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// INDEXING: Building the Search Index
// ═══════════════════════════════════════════════════════════════════════════════

// Index adds a document to the inverted index
//
// STEP-BY-STEP EXAMPLE:
// ----------------------
// Input: docID=1, document="The quick brown fox"
//
// Step 1: Tokenization
//
//	analyzer.Analyze() converts to: ["quick", "brown", "fox"]
//	(Note: "The" is removed as a stop word, and words are lowercased)
//
// Step 2: For each token, record its position
//
//	Token "quick" at position 0 in document 1
//	Token "brown" at position 1 in document 1
//	Token "fox"   at position 2 in document 1
//
// Step 3: Update the index
//
//	PostingsList["quick"] ← add Position{DocID:1, Offset:0}
//	PostingsList["brown"] ← add Position{DocID:1, Offset:1}
//	PostingsList["fox"]   ← add Position{DocID:1, Offset:2}
//
// Why record positions and not just document IDs?
// - Positions let us do phrase search ("brown fox" requires consecutive positions)
// - Positions let us rank by proximity (closer words = more relevant)
//
// Thread Safety Note:
// - We lock the entire indexing operation to prevent race conditions
// - If we didn't lock, two goroutines could corrupt the data structure
// ═══════════════════════════════════════════════════════════════════════════════
// BM25 INDEXING
// ═══════════════════════════════════════════════════════════════════════════════
// Index also enriches the index with BM25 statistics
//
// WHAT'S DIFFERENT WITH BM25:
// ---------------------------
// In addition to building the inverted index, we now track:
// 1. Document length (number of terms)
// 2. Term frequencies per document (how many times each term appears)
// 3. Total number of documents (for IDF calculation)
// 4. Total number of terms (for average document length)
//
// This metadata enables BM25 scoring later during search.
func (idx *InvertedIndex) Index(docID int, document string) {
	idx.mu.Lock()         // Acquire lock - only one goroutine can index at a time
	defer idx.mu.Unlock() // Release lock when function returns (even if it panics)

	slog.Info("indexing document", slog.Int("docID", docID))

	// STEP 1: Break document into searchable tokens
	// Example: "The Quick Brown Fox!" → ["quick", "brown", "fox"]
	tokens := Analyze(document)

	// STEP 2: Initialize document statistics
	docStats := DocumentStats{
		DocID:     docID,
		Length:    len(tokens),
		TermFreqs: make(map[string]int),
	}

	// STEP 3: Index each token and track term frequencies
	for position, token := range tokens {
		idx.indexToken(token, docID, position)
		docStats.TermFreqs[token]++
	}

	// STEP 4: Update global statistics
	idx.DocStats[docID] = docStats
	idx.TotalDocs++
	idx.TotalTerms += int64(len(tokens))
}

func (idx *InvertedIndex) UpsertDocument(docID int, document string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	_, existed := idx.DocStats[docID]
	if existed {
		idx.deleteDocumentUnlocked(docID)
		idx.indexDocumentUnlocked(docID, document)
	} else {
		idx.indexDocumentUnlocked(docID, document)
	}
}

// deleteDocumentUnlocked removes docID from postings, bitmaps, and BM25 stats.
// Caller must hold idx.mu. Returns false if the document was not indexed.
func (idx *InvertedIndex) deleteDocumentUnlocked(docID int) bool {
	stats, ok := idx.DocStats[docID]
	if !ok {
		return false
	}

	for term := range stats.TermFreqs {
		idx.removeTermPostingsForDoc(term, docID)
	}

	idx.TotalDocs--
	idx.TotalTerms -= int64(stats.Length)
	delete(idx.DocStats, docID)
	return true
}

// indexDocumentUnlocked adds a new document to the index. The caller must hold idx.mu.
// revision is stored in DocStats for synchronization; use 0 when not tracking versions.
func (idx *InvertedIndex) indexDocumentUnlocked(docID int, document string) {
	tokens := Analyze(document)

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

func (idx *InvertedIndex) removeTermPostingsForDoc(term string, docID int) {
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
		bm.Remove(uint32(docID))
		if bm.IsEmpty() {
			delete(idx.DocBitmaps, term)
		}
	}
}

// indexToken adds a single token occurrence to the index (HYBRID STORAGE)
//
// HOW IT WORKS:
// -------------
// 1. Update Roaring Bitmap (document-level)
//   - Set the bit for this document ID
//   - Enables fast document lookups and boolean operations
//   - Compressed storage (10-100x smaller than skip lists alone)
//
// 2. Update Skip List (position-level)
//   - Insert exact position (docID, offset)
//   - Enables phrase search and proximity ranking
//   - Maintains all position information
//
// 3. Best of both worlds!
//   - Fast document queries via bitmaps
//   - Detailed position queries via skip lists
//
// DocumentID and Offset are stored as ints
// - The SkipList uses sentinel values (BOF=MinInt, EOF=MaxInt) to mark boundaries
// - All position values are integers (no casting needed)
func (idx *InvertedIndex) indexToken(token string, docID, position int) {
	// STEP 1: Update roaring bitmap (document-level)
	// Create bitmap if this is the first time seeing this token
	if idx.DocBitmaps[token] == nil {
		idx.DocBitmaps[token] = roaring.NewBitmap()
	}
	// Set the bit for this document ID
	idx.DocBitmaps[token].Add(uint32(docID))

	// STEP 2: Update skip list (position-level)
	// Check if this token already has a posting list
	skipList, exists := idx.getPostingList(token)
	if !exists {
		// First time seeing this token - create a new SkipList
		skipList = *NewSkipList()
	}

	// Add this occurrence to the token's posting list
	skipList.Insert(Position{
		DocumentID: docID,    // Which document?
		Offset:     position, // Where in the document?
	})

	// Save the updated SkipList back to the map
	// (In Go, maps don't update automatically when you modify a struct value)
	idx.PostingsList[token] = skipList
}

// getPostingList retrieves the posting list for a token
//
// This is a simple helper to avoid repeating map lookup code.
// Returns (skipList, true) if found, (empty, false) if not found.
func (idx *InvertedIndex) getPostingList(token string) (SkipList, bool) {
	skipList, exists := idx.PostingsList[token]
	return skipList, exists
}

// ═══════════════════════════════════════════════════════════════════════════════
// BASIC SEARCH OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════
// These four methods (First, Last, Next, Previous) form the foundation of
// all search operations. Everything else is built on top of these primitives.
//
// Think of them like iterator operations:
// - First: Go to the beginning
// - Last: Go to the end
// - Next: Move forward
// - Previous: Move backward
// ═══════════════════════════════════════════════════════════════════════════════

// First returns the first occurrence of a token in the index
//
// EXAMPLE:
// --------
// Given: "quick" appears at [Doc1:Pos1, Doc3:Pos0, Doc5:Pos2]
// First("quick") returns Doc3:Pos0 (the earliest occurrence)
//
// Use case: Start searching for a token from the beginning
func (idx *InvertedIndex) First(token string) (Position, error) {
	skipList, exists := idx.getPostingList(token)
	if !exists {
		return EOFDocument, ErrNoPostingList
	}

	// The first position is at the bottom level (level 0) of the SkipList
	// The Head node points to the first real node via Tower[0]
	return skipList.Head.Tower[0].Key, nil
}

// Last returns the last occurrence of a token in the index
//
// EXAMPLE:
// --------
// Given: "quick" appears at [Doc1:Pos1, Doc3:Pos0, Doc5:Pos2]
// Last("quick") returns Doc5:Pos2 (the latest occurrence)
//
// Use case: Search backwards from the end
func (idx *InvertedIndex) Last(token string) (Position, error) {
	skipList, exists := idx.getPostingList(token)
	if !exists {
		return EOFDocument, ErrNoPostingList
	}

	// Traverse to the end of the SkipList
	return skipList.Last(), nil
}

// Next finds the next occurrence of a token after the given position
//
// EXAMPLE:
// --------
// Given: "brown" appears at [Doc1:Pos2, Doc3:Pos1, Doc3:Pos5, Doc5:Pos0]
// Next("brown", Doc3:Pos1) returns Doc3:Pos5
// Next("brown", Doc3:Pos5) returns Doc5:Pos0
// Next("brown", Doc5:Pos0) returns EOF (no more occurrences)
//
// Special cases:
// - If currentPos is BOF (beginning of file), return First
// - If currentPos is already EOF (end of file), stay at EOF
//
// Use case: Iterate through all occurrences of a word
func (idx *InvertedIndex) Next(token string, currentPos Position) (Position, error) {
	// Special case: Starting from the beginning
	if currentPos.IsBeginning() {
		return idx.First(token)
	}

	// Special case: Already at the end
	if currentPos.IsEnd() {
		return EOFDocument, nil
	}

	// Get the posting list for this token
	skipList, exists := idx.getPostingList(token)
	if !exists {
		return EOFDocument, ErrNoPostingList
	}

	// Find the next position after currentPos in the SkipList
	// FindGreaterThan returns the smallest position > currentPos
	nextPos, _ := skipList.FindGreaterThan(currentPos)
	return nextPos, nil
}

// Previous finds the previous occurrence of a token before the given position
//
// EXAMPLE:
// --------
// Given: "brown" appears at [Doc1:Pos2, Doc3:Pos1, Doc3:Pos5, Doc5:Pos0]
// Previous("brown", Doc5:Pos0) returns Doc3:Pos5
// Previous("brown", Doc3:Pos5) returns Doc3:Pos1
// Previous("brown", Doc1:Pos2) returns BOF (no earlier occurrences)
//
// Use case: Search backwards through occurrences
func (idx *InvertedIndex) Previous(token string, currentPos Position) (Position, error) {
	// Special case: Starting from the end
	if currentPos.IsEnd() {
		return idx.Last(token)
	}

	// Special case: Already at the beginning
	if currentPos.IsBeginning() {
		return BOFDocument, nil
	}

	// Get the posting list for this token
	skipList, exists := idx.getPostingList(token)
	if !exists {
		return BOFDocument, ErrNoPostingList
	}

	// Find the previous position before currentPos in the SkipList
	// FindLessThan returns the largest position < currentPos
	prevPos, _ := skipList.FindLessThan(currentPos)
	return prevPos, nil
}
