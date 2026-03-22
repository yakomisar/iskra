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
// PHRASE SEARCH: Finding Multi-Word Sequences
// ═══════════════════════════════════════════════════════════════════════════════
// Phrase search finds exact sequences of words.
//
// THE ALGORITHM:
// --------------
// To find "quick brown fox", we need three words at consecutive positions
// in the same document.
//
// Strategy:
// 1. Find ANY occurrence of all three words (might not be consecutive)
// 2. Walk backwards to find the start of the phrase
// 3. Check if the positions are consecutive
// 4. If yes, we found it! If no, try again from a different starting point
//
// VISUAL EXAMPLE:
// ---------------
// Document: "the quick brown dog ate the brown fox quickly"
// Positions:  0     1     2    3   4   5     6    7     8
//
// Searching for "brown fox":
//   Attempt 1:
//     - Find "brown" (any occurrence): Pos 2
//     - Find "fox" after Pos 2: Pos 7
//     - Walk back from "fox" to find "brown": Pos 6
//     - Check: Are Pos 6 and Pos 7 consecutive? YES! → Found it!
//
// ═══════════════════════════════════════════════════════════════════════════════

// NextPhrase finds the next occurrence of a phrase (sequence of words) in the index
//
// ALGORITHM WALKTHROUGH:
// ----------------------
// Query: "quick brown fox"
// StartPos: Beginning of file
//
// Step 1: Find the END of a potential phrase
//   - Find "quick" after startPos → maybe Doc2:Pos3
//   - Find "brown" after that     → maybe Doc2:Pos4
//   - Find "fox" after that       → maybe Doc2:Pos5
//   - endPos = Doc2:Pos5
//
// Step 2: Walk BACKWARDS to find the START
//   - From endPos, find previous "brown" → Doc2:Pos4
//   - From there, find previous "quick"  → Doc2:Pos3
//   - phraseStart = Doc2:Pos3
//
// Step 3: Validate it's a real phrase
//   - Same document? Yes (both Doc2)
//   - Consecutive positions? Yes (3, 4, 5)
//   - Distance = 5 - 3 = 2 (which equals 3 words - 1) ✓
//
// Step 4: If not valid, recurse from phraseStart
//   - This handles cases where words appear multiple times
//
// Why this algorithm?
// - It's efficient: We use the index to jump between occurrences
// - It handles multiple occurrences: Recursion keeps searching
// - It validates correctness: We check for consecutive positions
func (idx *InvertedIndex) NextPhrase(query string, startPos Position) []Position {
	terms := strings.Fields(query) // Split "quick brown fox" → ["quick", "brown", "fox"]

	// STEP 1: Find the end of a potential phrase match
	endPos := idx.findPhraseEnd(terms, startPos)
	if endPos.IsEnd() {
		// No more occurrences of all words exist
		return []Position{EOFDocument, EOFDocument}
	}

	// STEP 2: Walk backwards to find where the phrase starts
	phraseStart := idx.findPhraseStart(terms, endPos)

	// STEP 3: Validate that we found a real consecutive phrase
	if idx.isValidPhrase(phraseStart, endPos, len(terms)) {
		// Success! Return [start, end] positions of the phrase
		return []Position{phraseStart, endPos}
	}

	// STEP 4: Not a valid phrase - try again from the start position
	// This handles cases like: "brown dog brown fox" when searching for "brown fox"
	return idx.NextPhrase(query, phraseStart)
}

// findPhraseEnd locates the ending position of a potential phrase
//
// HOW IT WORKS:
// -------------
// Starting from startPos, we "hop" through the document finding each word.
//
// Example: Finding "quick brown fox" starting from Doc1:Pos0
//
//	Step 1: currentPos = Doc1:Pos0
//	Step 2: Find "quick" after Doc1:Pos0 → currentPos = Doc1:Pos3
//	Step 3: Find "brown" after Doc1:Pos3 → currentPos = Doc1:Pos4
//	Step 4: Find "fox" after Doc1:Pos4   → currentPos = Doc1:Pos5
//	Return: Doc1:Pos5 (position of the last word "fox")
//
// If any word isn't found, we return EOF (no phrase exists)
func (idx *InvertedIndex) findPhraseEnd(terms []string, startPos Position) Position {
	currentPos := startPos

	// For each word in the phrase, find its next occurrence
	for _, term := range terms {
		currentPos, _ = idx.Next(term, currentPos)

		// If we can't find this word, the phrase doesn't exist
		if currentPos.IsEnd() {
			return EOFDocument
		}
	}

	// currentPos now points to the last word of the phrase
	return currentPos
}

// findPhraseStart walks backward to find where the phrase begins
//
// HOW IT WORKS:
// -------------
// Starting from the END position, we walk backwards through the phrase.
//
// Example: We found "fox" at Doc1:Pos5, now find the start of "quick brown fox"
//
//	Step 1: currentPos = Doc1:Pos5 (we're at "fox")
//	Step 2: Find "brown" before Doc1:Pos5 → currentPos = Doc1:Pos4
//	Step 3: Find "quick" before Doc1:Pos4 → currentPos = Doc1:Pos3
//	Return: Doc1:Pos3 (position of the first word "quick")
//
// Why skip the last word?
// - We already know where the last word is (at endPos)
// - We only need to walk back through the first N-1 words
func (idx *InvertedIndex) findPhraseStart(terms []string, endPos Position) Position {
	currentPos := endPos

	// Walk backwards through all words EXCEPT the last one
	// (we already know the last word's position - it's endPos)
	for i := len(terms) - 2; i >= 0; i-- {
		currentPos, _ = idx.Previous(terms[i], currentPos)
	}

	// currentPos now points to the first word of the phrase
	return currentPos
}

// isValidPhrase checks if positions form a valid consecutive phrase
//
// VALIDATION RULES:
// -----------------
// For a valid phrase, we need:
// 1. All words in the SAME document
// 2. Words at CONSECUTIVE positions
//
// Example: Checking "quick brown fox" (3 words)
//
//	start = Doc1:Pos3
//	end = Doc1:Pos5
//
//	Check 1: Same document? Doc1 == Doc1 ✓
//	Check 2: Consecutive? (5 - 3) == (3 - 1) → 2 == 2 ✓
//	Result: VALID
//
// Counter-example: NOT a valid phrase
//
//	start = Doc1:Pos3
//	end = Doc1:Pos7
//
//	Check 2: Consecutive? (7 - 3) == (3 - 1) → 4 == 2 ✗
//	Result: INVALID (there are extra words between them)
func (idx *InvertedIndex) isValidPhrase(start, end Position, termCount int) bool {
	// Calculate expected distance for consecutive words
	// For 3 words, positions should be like [0,1,2] → distance = 2
	expectedDistance := termCount - 1

	// Calculate actual distance between start and end
	actualDistance := end.GetOffset() - start.GetOffset()

	// Both conditions must be true
	return start.DocumentID == end.DocumentID && actualDistance == expectedDistance
}

// FindAllPhrases finds ALL occurrences of a phrase in the entire index
//
// ALGORITHM:
// ----------
// This is just a loop that repeatedly calls NextPhrase until we reach EOF.
//
// Example: Finding all occurrences of "brown fox"
//
//	Iteration 1:
//	  - Search from BOF → Found at Doc2:Pos[3-4]
//	  - Add to results
//	  - Continue from Doc2:Pos3
//
//	Iteration 2:
//	  - Search from Doc2:Pos3 → Found at Doc5:Pos[1-2]
//	  - Add to results
//	  - Continue from Doc5:Pos1
//
//	Iteration 3:
//	  - Search from Doc5:Pos1 → Returns EOF
//	  - Stop searching
//
// Result: [[Doc2:Pos3-4], [Doc5:Pos1-2]]
func (idx *InvertedIndex) FindAllPhrases(query string, startPos Position) [][]Position {
	var allMatches [][]Position
	currentPos := BOFDocument // Start from the beginning

	// Keep searching until we reach the end of file
	for !currentPos.IsEnd() {
		// Find the next occurrence of the phrase
		phrasePositions := idx.NextPhrase(query, currentPos)
		phraseStart := phrasePositions[0]

		// If we found a valid phrase (not EOF), add it to results
		if !phraseStart.IsEnd() {
			allMatches = append(allMatches, phrasePositions)
		}

		// Move to where we found the phrase to continue searching
		currentPos = phraseStart
	}

	return allMatches
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROXIMITY SEARCH: Finding Documents Containing All Terms
// ═══════════════════════════════════════════════════════════════════════════════
// A "cover" is a range of positions that contains ALL search terms.
// Unlike phrase search, the words don't need to be consecutive or in order.
//
// EXAMPLE:
// --------
// Document: "the quick brown dog jumped over the lazy fox"
// Positions:  0     1     2    3     4      5    6    7    8
//
// Searching for ["quick", "fox"]:
//   Cover 1: Pos 1 to Pos 8 (entire range containing both words)
//   This is the MINIMAL cover (smallest range containing all terms)
//
// WHY USE COVERS?
// ---------------
// Covers are used for:
// 1. Boolean search: Find documents with ALL terms (AND query)
// 2. Proximity ranking: Closer terms = higher relevance
// 3. Snippet generation: Show the most relevant part of a document
//
// THE ALGORITHM:
// --------------
// To find a cover:
// 1. Find the FURTHEST occurrence of any term (this is the cover end)
// 2. Walk BACKWARDS to find the EARLIEST occurrence of each term
// 3. Check if all terms are in the same document
// 4. If yes, we found a cover! If no, try again.
// ═══════════════════════════════════════════════════════════════════════════════

// NextCover finds the next "cover" - a range containing all given tokens
//
// ALGORITHM WALKTHROUGH:
// ----------------------
// Query: ["quick", "fox", "brown"]
// StartPos: Beginning of file
//
// PHASE 1: Find the cover END (furthest position)
//   - Find "quick" after startPos → maybe Doc2:Pos1
//   - Find "fox" after startPos   → maybe Doc2:Pos8  ← furthest
//   - Find "brown" after startPos → maybe Doc2:Pos2
//   - coverEnd = Doc2:Pos8
//
// PHASE 2: Find the cover START (earliest position before end)
//   - From Doc2:Pos9, find previous "quick" → Doc2:Pos1  ← earliest
//   - From Doc2:Pos9, find previous "fox"   → Doc2:Pos8
//   - From Doc2:Pos9, find previous "brown" → Doc2:Pos2
//   - coverStart = Doc2:Pos1
//
// PHASE 3: Validate the cover
//   - Same document? Yes (all in Doc2) ✓
//   - Return [Doc2:Pos1, Doc2:Pos8]
//
// If not same document, recurse from coverStart to find the next cover.
//
// Why this algorithm?
// - Greedy approach: We find the furthest occurrence first
// - Efficient: Uses index jumps instead of scanning
// - Minimal covers: Always finds the smallest valid range
func (idx *InvertedIndex) NextCover(tokens []string, startPos Position) []Position {
	// PHASE 1: Find the END of the cover (furthest position)
	coverEnd := idx.findCoverEnd(tokens, startPos)
	if coverEnd.IsEnd() {
		// Can't find all tokens - no cover exists
		return []Position{EOFDocument, EOFDocument}
	}

	// PHASE 2: Find the START of the cover (earliest position)
	coverStart := idx.findCoverStart(tokens, coverEnd)

	// PHASE 3: Validate the cover
	if coverStart.DocumentID == coverEnd.DocumentID {
		// Success! All tokens are in the same document
		return []Position{coverStart, coverEnd}
	}

	// Tokens span multiple documents - try again from coverStart
	return idx.NextCover(tokens, coverStart)
}

// findCoverEnd finds the furthest position among all tokens
//
// HOW IT WORKS:
// -------------
// We find the next occurrence of EACH token and track the furthest one.
//
// Example: Finding cover end for ["quick", "brown", "fox"] from Doc1:Pos0
//
//	Step 1: Find "quick" after Pos0 → Doc2:Pos1
//	        maxPos = Doc2:Pos1
//
//	Step 2: Find "brown" after Pos0 → Doc2:Pos2
//	        Is Doc2:Pos2 after Doc2:Pos1? Yes
//	        maxPos = Doc2:Pos2
//
//	Step 3: Find "fox" after Pos0 → Doc2:Pos8
//	        Is Doc2:Pos8 after Doc2:Pos2? Yes
//	        maxPos = Doc2:Pos8
//
//	Return: Doc2:Pos8 (the furthest position)
//
// Special case: If ANY token returns EOF, we can't form a cover
func (idx *InvertedIndex) findCoverEnd(tokens []string, startPos Position) Position {
	maxPos := startPos

	for _, token := range tokens {
		// Find next occurrence of this token
		tokenPos, _ := idx.Next(token, startPos)

		// If any token is not found, we can't create a cover
		if tokenPos.IsEnd() {
			return EOFDocument
		}

		// Keep track of the furthest position
		if tokenPos.IsAfter(maxPos) {
			maxPos = tokenPos
		}
	}

	return maxPos
}

// findCoverStart finds the earliest position that still covers all tokens
//
// HOW IT WORKS:
// -------------
// Starting from just after the cover end, we walk backwards to find each token.
//
// Example: Finding cover start for ["quick", "brown", "fox"]
//
//	       with coverEnd at Doc2:Pos8
//
//	searchBound = Doc2:Pos9 (one position after the end)
//
//	Step 1: Find "quick" before Pos9 → Doc2:Pos1  ← earliest so far
//	        minPos = Doc2:Pos1
//
//	Step 2: Find "brown" before Pos9 → Doc2:Pos2
//	        Is Doc2:Pos2 before Doc2:Pos1? No
//	        minPos stays Doc2:Pos1
//
//	Step 3: Find "fox" before Pos9 → Doc2:Pos8
//	        Is Doc2:Pos8 before Doc2:Pos1? No
//	        minPos stays Doc2:Pos1
//
//	Return: Doc2:Pos1 (the earliest position)
//
// Why search from (endPos + 1)?
// - Previous() returns positions STRICTLY BEFORE the search point
// - By searching from endPos+1, we can find tokens AT endPos
func (idx *InvertedIndex) findCoverStart(tokens []string, endPos Position) Position {
	minPos := BOFDocument

	// Create a search bound just after the cover end
	// This ensures we can find tokens AT the end position
	searchBound := Position{
		DocumentID: endPos.DocumentID,
		Offset:     endPos.Offset + 1,
	}

	for _, token := range tokens {
		// Find the previous occurrence of this token before searchBound
		tokenPos, _ := idx.Previous(token, searchBound)

		// Keep track of the earliest position
		if minPos.IsBeginning() || tokenPos.IsBefore(minPos) {
			minPos = tokenPos
		}
	}

	return minPos
}

// ═══════════════════════════════════════════════════════════════════════════════
// RANKING: Scoring Search Results by Relevance
// ═══════════════════════════════════════════════════════════════════════════════
// Not all search results are equally relevant. We need to rank them!
//
// PROXIMITY RANKING:
// ------------------
// The idea: Documents where search terms appear CLOSER together are more relevant.
//
// Example: Searching for "machine learning"
//   Doc A: "machine learning is..."        (distance: 1) → HIGH score
//   Doc B: "machine ... learning"          (distance: 3) → MEDIUM score
//   Doc C: "machine ... ... ... learning"  (distance: 5) → LOW score
//
// SCORING FORMULA:
// ----------------
// For each cover in a document:
//   score += 1 / (coverEnd - coverStart + 1)
//
// Why this formula?
// - Smaller distances → larger scores (inversely proportional)
// - Multiple covers → higher score (sum of all covers)
// - Simple and fast to compute
//
// EXAMPLE CALCULATION:
// --------------------
// Document: "quick brown fox jumped over quick brown dog"
// Positions:   0      1     2     3      4     5      6    7
//
// Searching for ["quick", "brown"]:
//   Cover 1: Pos[0-1] → score += 1/(1-0+1) = 1/2 = 0.5
//   Cover 2: Pos[5-6] → score += 1/(6-5+1) = 1/2 = 0.5
//   Total score: 1.0
//
// A document with terms closer together:
// Document: "quick brown"
// Positions:   0      1
//   Cover 1: Pos[0-1] → score = 1/(1-0+1) = 1/2 = 0.5
//   Total score: 0.5 (but only ONE occurrence)
//
// ═══════════════════════════════════════════════════════════════════════════════

// Match represents a search result with its positions and relevance score
//
// STRUCTURE:
// ----------
// Offsets: The [start, end] positions of a cover in a document
// Score: The relevance score (higher = more relevant)
//
// Example Match:
//
//	Offsets: [Doc3:Pos1, Doc3:Pos5]  ← This document matches from Pos1 to Pos5
//	Score: 2.7                         ← Relevance score
type Match struct {
	DocID   int        // Document identifier
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
//
// IDF FORMULA:
// ------------
// IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)
//
// Where:
//
//	N = total number of documents
//	df = number of documents containing the term
//
// INTUITION:
// ----------
// - Rare terms (low df) get high IDF scores
// - Common terms (high df) get low IDF scores
// - This makes rare terms more important for ranking
//
// EXAMPLE:
// --------
// Total docs: 1000
// Term "the": appears in 950 docs → IDF ≈ 0.05 (very common, low importance)
// Term "quantum": appears in 5 docs → IDF ≈ 5.3 (rare, high importance)
//
// PERFORMANCE BOOST WITH ROARING BITMAPS:
// ----------------------------------------
// Instead of traversing skip lists to count documents, we use bitmaps:
// - Old way: O(n) traverse skip list, count unique docs
// - New way: O(1) bitmap.GetCardinality()
// This is 10-100x faster for common terms!
func (idx *InvertedIndex) calculateIDF(term string) float64 {
	// Use roaring bitmap for instant document count
	bitmap, exists := idx.DocBitmaps[term]
	if !exists {
		return 0.0
	}

	// Get document frequency instantly from bitmap cardinality
	df := float64(bitmap.GetCardinality())

	if df == 0 {
		return 0.0
	}

	N := float64(idx.TotalDocs)

	// BM25 IDF formula (with smoothing to avoid negative values)
	return math.Log((N-df+0.5)/(df+0.5) + 1.0)
}

// countDocsInPostingList counts unique documents in a posting list
func (idx *InvertedIndex) countDocsInPostingList(skipList SkipList) int {
	uniqueDocs := make(map[int]bool)

	current := skipList.Head.Tower[0]
	for current != nil {
		docID := current.Key.GetDocumentID()
		uniqueDocs[docID] = true
		current = current.Tower[0]
	}

	return len(uniqueDocs)
}

// calculateBM25Score computes the BM25 score for a document given query terms
//
// BM25 ALGORITHM:
// ---------------
//  1. For each query term:
//     a. Calculate IDF(term) - how rare is this term?
//     b. Get term frequency in document
//     c. Apply saturation and length normalization
//     d. Accumulate score
//
// EXAMPLE CALCULATION:
// --------------------
// Query: "machine learning"
// Doc 5: 200 words, "machine" appears 3 times, "learning" appears 2 times
// Corpus: 1000 docs, avg length 150 words
//
// For "machine" (appears in 100 docs):
//
//	IDF = log((1000 - 100 + 0.5) / (100 + 0.5) + 1) ≈ 2.3
//	TF = 3
//	normalized_TF = (3 * 2.5) / (3 + 1.5 * (1 - 0.75 + 0.75 * (200/150)))
//	              = 7.5 / 5.0 ≈ 1.5
//	score += 2.3 * 1.5 ≈ 3.45
//
// For "learning" (appears in 50 docs):
//
//	IDF ≈ 2.9
//	TF = 2
//	normalized_TF ≈ 1.2
//	score += 2.9 * 1.2 ≈ 3.48
//
// Total BM25 score: 3.45 + 3.48 = 6.93
func (idx *InvertedIndex) calculateBM25Score(docID int, queryTerms []string) float64 {
	docStats, exists := idx.DocStats[docID]
	if !exists {
		return 0.0
	}

	// Calculate average document length
	avgDocLen := float64(idx.TotalTerms) / float64(idx.TotalDocs)
	docLen := float64(docStats.Length)

	score := 0.0
	k1 := idx.BM25Params.K1
	b := idx.BM25Params.B

	// Process each query term
	for _, term := range queryTerms {
		// Get IDF for this term
		idf := idx.calculateIDF(term)

		// Get term frequency in this document
		tf := float64(docStats.TermFreqs[term])

		if tf > 0 {
			// BM25 formula with length normalization
			numerator := tf * (k1 + 1)
			denominator := tf + k1*(1-b+b*(docLen/avgDocLen))
			score += idf * (numerator / denominator)
		}
	}

	return score
}

// RankBM25 performs BM25 ranking of search results
//
// ALGORITHM:
// ----------
// 1. Tokenize query
// 2. Find all documents containing at least one query term
// 3. Calculate BM25 score for each document
// 4. Sort by score (descending)
// 5. Return top K results
//
// EXAMPLE:
// --------
// Query: "machine learning algorithms"
//
// Step 1: Tokenize → ["machine", "learning", "algorithms"]
//
// Step 2: Find candidate documents:
//
//	"machine" appears in: [Doc1, Doc3, Doc5, Doc7]
//	"learning" appears in: [Doc1, Doc2, Doc5]
//	"algorithms" appears in: [Doc2, Doc5, Doc8]
//	Candidates: [Doc1, Doc2, Doc3, Doc5, Doc7, Doc8]
//
// Step 3: Calculate BM25 scores:
//
//	Doc1: 12.5 (has "machine" and "learning")
//	Doc2: 8.3 (has "learning" and "algorithms")
//	Doc3: 3.2 (only has "machine")
//	Doc5: 15.7 (has all three terms!)
//	Doc7: 2.1 (only has "machine")
//	Doc8: 4.5 (only has "algorithms")
//
// Step 4: Sort: [Doc5, Doc1, Doc2, Doc8, Doc3, Doc7]
//
// Step 5: Return top 3: [Doc5, Doc1, Doc2]
func (idx *InvertedIndex) RankBM25(query string, maxResults int) []Match {
	slog.Info("BM25 ranking", slog.String("query", query))

	tokens := Analyze(query)
	if len(tokens) == 0 {
		return []Match{}
	}

	slog.Info("search tokens", slog.String("tokens", fmt.Sprintf("%v", tokens)))

	// Find all candidate documents (documents containing at least one query term)
	candidates := idx.findCandidateDocuments(tokens)

	// Calculate BM25 score for each candidate
	results := make([]Match, 0, len(candidates))
	for docID := range candidates {
		score := idx.calculateBM25Score(docID, tokens)

		if score > 0 {
			results = append(results, Match{
				DocID:   docID,
				Offsets: candidates[docID], // Positions where terms appear
				Score:   score,
			})
		}
	}

	// Sort by score (descending)
	idx.sortMatchesByScore(results)

	// Return top K results
	return limitResults(results, maxResults)
}

// findCandidateDocuments finds all documents containing at least one query term
//
// Returns a map: DocID → Positions where query terms appear
//
// PERFORMANCE BOOST WITH ROARING BITMAPS:
// ----------------------------------------
// We use a two-phase approach:
// 1. Fast filtering: Use bitmaps to find candidate document IDs (O(1) per term)
// 2. Position lookup: Only fetch positions for candidate documents
//
// OLD APPROACH: Traverse every skip list node (slow)
// NEW APPROACH: Bitmap union + targeted position lookup (fast!)
func (idx *InvertedIndex) findCandidateDocuments(tokens []string) map[int][]Position {
	candidates := make(map[int][]Position)

	// PHASE 1: Use bitmaps to quickly find all candidate document IDs
	candidateDocs := make(map[int]bool)
	for _, token := range tokens {
		bitmap, exists := idx.DocBitmaps[token]
		if !exists {
			continue
		}

		// Iterate through document IDs in the bitmap
		iter := bitmap.Iterator()
		for iter.HasNext() {
			docID := int(iter.Next())
			candidateDocs[docID] = true
		}
	}

	// PHASE 2: For each candidate document, fetch positions from skip lists
	// This is still needed for BM25 scoring (we need exact positions)
	for _, token := range tokens {
		skipList, exists := idx.getPostingList(token)
		if !exists {
			continue
		}

		// Only traverse skip list for positions in candidate documents
		current := skipList.Head.Tower[0]
		for current != nil {
			docID := current.Key.GetDocumentID()
			// Only add if this is a candidate document
			if candidateDocs[docID] {
				candidates[docID] = append(candidates[docID], current.Key)
			}
			current = current.Tower[0]
		}
	}

	return candidates
}

// sortMatchesByScore sorts matches by score in descending order (higher scores first)
func (idx *InvertedIndex) sortMatchesByScore(matches []Match) {
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Score > matches[j].Score
	})
}

// RankProximity performs proximity-based ranking of search results
//
// THIS IS THE MAIN SEARCH FUNCTION!
//
// COMPLETE EXAMPLE:
// -----------------
// Query: "machine learning"
// MaxResults: 10
//
// Step 1: Tokenize query
//
//	"machine learning" → ["machine", "learning"]
//
// Step 2: Find all covers (ranges containing both words)
//
//	Doc1: Cover[0-1], Cover[5-6]    → score = 0.5 + 0.5 = 1.0
//	Doc2: Cover[0-5]                → score = 0.167
//	Doc3: Cover[2-3], Cover[10-11]  → score = 0.5 + 0.5 = 1.0
//	Doc4: Cover[1-1]                → Wait, both words at same position? Impossible!
//	                                   (This means one word appears twice)
//
// Step 3: Return top 10 results
//
//	Result: [Doc1, Doc3, Doc2] (sorted by score, limited to 10)
//
// ALGORITHM WALKTHROUGH:
// ----------------------
// We iterate through ALL covers in the index, accumulating scores per document.
//
// Iteration 1: Find first cover → Doc1:Pos[0-1]
//   - New document Doc1 detected
//   - Calculate score: 1/(1-0+1) = 0.5
//   - Current document: Doc1, current score: 0.5
//
// Iteration 2: Find next cover → Doc1:Pos[5-6]
//   - Still in Doc1 (not a new document)
//   - Add to score: 0.5 + 1/(6-5+1) = 1.0
//   - Current document: Doc1, current score: 1.0
//
// Iteration 3: Find next cover → Doc2:Pos[0-5]
//   - New document Doc2 detected!
//   - Save previous: Match{Doc1, score=1.0}
//   - Start new: Doc2, score = 1/(5-0+1) = 0.167
//
// ... continue until EOF ...
//
// Final step: Return top K results
func (idx *InvertedIndex) RankProximity(query string, maxResults int) []Match {
	slog.Info("proximity ranking", slog.String("query", query))

	// STEP 1: Tokenize the query (same as indexing)
	tokens := Analyze(query)
	if len(tokens) == 0 {
		// Empty query → no results
		return []Match{}
	}

	slog.Info("search tokens", slog.String("tokens", fmt.Sprintf("%v", tokens)))

	// STEP 2: Find and score all covers
	results := idx.collectProximityMatches(tokens)

	// STEP 3: Limit to top K results
	return limitResults(results, maxResults)
}

// collectProximityMatches finds and scores all proximity matches
//
// This is the core ranking loop that:
// 1. Finds all covers
// 2. Groups them by document
// 3. Calculates cumulative scores per document
//
// STATE TRACKING:
// ---------------
// We maintain state across iterations:
// - currentCandidate: The [start, end] positions of the current document's match
// - currentScore: The accumulated score for the current document
// - matches: The final list of all document matches
//
// TRANSITION DETECTION:
// ---------------------
// When we find a cover in a NEW document:
//
//	→ Save the current document's match
//	→ Reset state for the new document
func (idx *InvertedIndex) collectProximityMatches(tokens []string) []Match {
	var matches []Match

	// Find the first cover to initialize our state
	coverPositions := idx.NextCover(tokens, BOFDocument)
	coverStart, coverEnd := coverPositions[0], coverPositions[1]

	// Initialize tracking variables
	currentCandidate := []Position{coverStart, coverEnd}
	currentScore := 0.0

	// Loop through all covers until we reach EOF
	for !coverStart.IsEnd() {
		// DETECTION: Did we move to a new document?
		if currentCandidate[0].DocumentID < coverStart.DocumentID {
			// Yes! Save the previous document's match
			matches = append(matches, Match{
				Offsets: currentCandidate,
				Score:   currentScore,
			})

			// Reset state for the new document
			currentCandidate = []Position{coverStart, coverEnd}
			currentScore = 0
		}

		// SCORING: Calculate proximity score for this cover
		// Formula: 1 / (distance + 1)
		// - Smaller distance → higher score
		// - +1 to avoid division by zero when start==end
		proximity := float64(coverEnd.Offset - coverStart.Offset + 1)
		currentScore += 1 / proximity

		// Find the next cover
		coverPositions = idx.NextCover(tokens, coverStart)
		coverStart, coverEnd = coverPositions[0], coverPositions[1]
	}

	// Don't forget the last document!
	// When we reach EOF, we still have one unsaved match
	if !currentCandidate[0].IsEnd() {
		matches = append(matches, Match{
			Offsets: currentCandidate,
			Score:   currentScore,
		})
	}

	return matches
}

// limitResults returns at most maxResults items
//
// Simple helper to truncate the results list.
// Uses math.Min to avoid index-out-of-bounds errors.
//
// Example:
//
//	matches = [Match1, Match2, Match3, Match4, Match5]
//	maxResults = 3
//	Returns: [Match1, Match2, Match3]
func limitResults(matches []Match, maxResults int) []Match {
	limit := int(math.Min(float64(maxResults), float64(len(matches))))
	return matches[:limit]
}
