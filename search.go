package iskra

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"
)

// ═══════════════════════════════════════════════════════════════════════════════
// PHRASE SEARCH
// ═══════════════════════════════════════════════════════════════════════════════

// NextPhrase finds the next occurrence of a phrase (sequence of words) in the index
func (idx *InvertedIndex) NextPhrase(query string, startPos Position) []Position {
	terms := strings.Fields(query)

	endPos := idx.findPhraseEnd(terms, startPos)
	if endPos.IsEnd() {
		return []Position{EOFDocument, EOFDocument}
	}

	phraseStart := idx.findPhraseStart(terms, endPos)

	if idx.isValidPhrase(phraseStart, endPos, len(terms)) {
		return []Position{phraseStart, endPos}
	}

	return idx.NextPhrase(query, phraseStart)
}

func (idx *InvertedIndex) findPhraseEnd(terms []string, startPos Position) Position {
	currentPos := startPos
	for _, term := range terms {
		currentPos, _ = idx.Next(term, currentPos)
		if currentPos.IsEnd() {
			return EOFDocument
		}
	}
	return currentPos
}

func (idx *InvertedIndex) findPhraseStart(terms []string, endPos Position) Position {
	currentPos := endPos
	for i := len(terms) - 2; i >= 0; i-- {
		currentPos, _ = idx.Previous(terms[i], currentPos)
	}
	return currentPos
}

func (idx *InvertedIndex) isValidPhrase(start, end Position, termCount int) bool {
	expectedDistance := termCount - 1
	actualDistance := end.GetOffset() - start.GetOffset()
	return start.DocumentID == end.DocumentID && actualDistance == expectedDistance
}

// FindAllPhrases finds ALL occurrences of a phrase in the entire index
func (idx *InvertedIndex) FindAllPhrases(query string, startPos Position) [][]Position {
	var allMatches [][]Position
	currentPos := BOFDocument

	for !currentPos.IsEnd() {
		phrasePositions := idx.NextPhrase(query, currentPos)
		phraseStart := phrasePositions[0]

		if !phraseStart.IsEnd() {
			allMatches = append(allMatches, phrasePositions)
		}

		currentPos = phraseStart
	}

	return allMatches
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROXIMITY SEARCH
// ═══════════════════════════════════════════════════════════════════════════════

// NextCover finds the next "cover" - a range containing all given tokens
func (idx *InvertedIndex) NextCover(tokens []string, startPos Position) []Position {
	coverEnd := idx.findCoverEnd(tokens, startPos)
	if coverEnd.IsEnd() {
		return []Position{EOFDocument, EOFDocument}
	}

	coverStart := idx.findCoverStart(tokens, coverEnd)

	if coverStart.DocumentID == coverEnd.DocumentID {
		return []Position{coverStart, coverEnd}
	}

	return idx.NextCover(tokens, coverStart)
}

func (idx *InvertedIndex) findCoverEnd(tokens []string, startPos Position) Position {
	maxPos := startPos
	for _, token := range tokens {
		tokenPos, _ := idx.Next(token, startPos)
		if tokenPos.IsEnd() {
			return EOFDocument
		}
		if tokenPos.IsAfter(maxPos) {
			maxPos = tokenPos
		}
	}
	return maxPos
}

func (idx *InvertedIndex) findCoverStart(tokens []string, endPos Position) Position {
	minPos := BOFDocument
	searchBound := Position{
		DocumentID: endPos.DocumentID,
		Offset:     endPos.Offset + 1,
	}

	for _, token := range tokens {
		tokenPos, _ := idx.Previous(token, searchBound)
		if minPos.IsBeginning() || tokenPos.IsBefore(minPos) {
			minPos = tokenPos
		}
	}
	return minPos
}

// ═══════════════════════════════════════════════════════════════════════════════
// RANKING
// ═══════════════════════════════════════════════════════════════════════════════

// Match represents a search result with its positions and relevance score.
// DocID is uint32 to match the index-wide convention.
type Match struct {
	DocID   uint32     // Document identifier
	Offsets []Position // Where the match was found [start, end]
	Score   float64    // How relevant is this match?
}

// GetKey generates a unique identifier for the match
func (m *Match) GetKey() (string, error) {
	data, err := json.Marshal(m.DocID)
	if err != nil {
		return "", err
	}
	hash := md5.Sum(data)
	return hex.EncodeToString(hash[:]), nil
}

// calculateIDF computes the Inverse Document Frequency for a term
func (idx *InvertedIndex) calculateIDF(term string) float64 {
	bitmap, exists := idx.DocBitmaps[term]
	if !exists {
		return 0.0
	}

	df := float64(bitmap.GetCardinality())
	if df == 0 {
		return 0.0
	}

	N := float64(idx.TotalDocs)
	return math.Log((N-df+0.5)/(df+0.5) + 1.0)
}

// countDocsInPostingList counts unique documents in a posting list
func (idx *InvertedIndex) countDocsInPostingList(skipList SkipList) int {
	uniqueDocs := make(map[uint32]bool)
	current := skipList.Head.Tower[0]
	for current != nil {
		uniqueDocs[current.Key.GetDocumentID()] = true
		current = current.Tower[0]
	}
	return len(uniqueDocs)
}

// calculateBM25Score computes the BM25 score for a document given query terms
func (idx *InvertedIndex) calculateBM25Score(docID uint32, queryTerms []string) float64 {
	docStats, exists := idx.DocStats[docID]
	if !exists {
		return 0.0
	}

	avgDocLen := float64(idx.TotalTerms) / float64(idx.TotalDocs)
	docLen := float64(docStats.Length)

	score := 0.0
	k1 := idx.BM25Params.K1
	b := idx.BM25Params.B

	for _, term := range queryTerms {
		idf := idx.calculateIDF(term)
		tf := float64(docStats.TermFreqs[term])
		if tf > 0 {
			numerator := tf * (k1 + 1)
			denominator := tf + k1*(1-b+b*(docLen/avgDocLen))
			score += idf * (numerator / denominator)
		}
	}
	return score
}

// RankBM25 performs BM25 ranking of search results
func (idx *InvertedIndex) RankBM25(query string, maxResults int) []Match {
	slog.Info("BM25 ranking", slog.String("query", query))

	tokens := Analyze(query)
	if len(tokens) == 0 {
		return []Match{}
	}

	slog.Info("search tokens", slog.String("tokens", fmt.Sprintf("%v", tokens)))

	candidates := idx.findCandidateDocuments(tokens)

	results := make([]Match, 0, len(candidates))
	for docID := range candidates {
		score := idx.calculateBM25Score(docID, tokens)
		if score > 0 {
			results = append(results, Match{
				DocID:   docID,
				Offsets: candidates[docID],
				Score:   score,
			})
		}
	}

	idx.sortMatchesByScore(results)
	return limitResults(results, maxResults)
}

// findCandidateDocuments finds all documents containing at least one query term
func (idx *InvertedIndex) findCandidateDocuments(tokens []string) map[uint32][]Position {
	candidates := make(map[uint32][]Position)

	// Phase 1: bitmap fast-path to collect candidate doc IDs
	candidateDocs := make(map[uint32]bool)
	for _, token := range tokens {
		bitmap, exists := idx.DocBitmaps[token]
		if !exists {
			continue
		}
		iter := bitmap.Iterator()
		for iter.HasNext() {
			candidateDocs[iter.Next()] = true
		}
	}

	// Phase 2: collect positions from skip lists for candidate docs only
	for _, token := range tokens {
		skipList, exists := idx.getPostingList(token)
		if !exists {
			continue
		}
		current := skipList.Head.Tower[0]
		for current != nil {
			docID := current.Key.GetDocumentID()
			if candidateDocs[docID] {
				candidates[docID] = append(candidates[docID], current.Key)
			}
			current = current.Tower[0]
		}
	}

	return candidates
}

// sortMatchesByScore sorts matches by score in descending order
func (idx *InvertedIndex) sortMatchesByScore(matches []Match) {
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Score > matches[j].Score
	})
}

// RankProximity performs proximity-based ranking of search results
func (idx *InvertedIndex) RankProximity(query string, maxResults int) []Match {
	slog.Info("proximity ranking", slog.String("query", query))

	tokens := Analyze(query)
	if len(tokens) == 0 {
		return []Match{}
	}

	slog.Info("search tokens", slog.String("tokens", fmt.Sprintf("%v", tokens)))

	results := idx.collectProximityMatches(tokens)
	return limitResults(results, maxResults)
}

// collectProximityMatches finds and scores all proximity matches
func (idx *InvertedIndex) collectProximityMatches(tokens []string) []Match {
	var matches []Match

	coverPositions := idx.NextCover(tokens, BOFDocument)
	coverStart, coverEnd := coverPositions[0], coverPositions[1]

	currentCandidate := []Position{coverStart, coverEnd}
	currentScore := 0.0

	for !coverStart.IsEnd() {
		if currentCandidate[0].DocumentID < coverStart.DocumentID {
			matches = append(matches, Match{
				Offsets: currentCandidate,
				Score:   currentScore,
			})
			currentCandidate = []Position{coverStart, coverEnd}
			currentScore = 0
		}

		proximity := float64(coverEnd.Offset - coverStart.Offset + 1)
		currentScore += 1 / proximity

		coverPositions = idx.NextCover(tokens, coverStart)
		coverStart, coverEnd = coverPositions[0], coverPositions[1]
	}

	if !currentCandidate[0].IsEnd() {
		matches = append(matches, Match{
			Offsets: currentCandidate,
			Score:   currentScore,
		})
	}

	return matches
}

// limitResults returns at most maxResults items
func limitResults(matches []Match, maxResults int) []Match {
	limit := int(math.Min(float64(maxResults), float64(len(matches))))
	return matches[:limit]
}
