package iskra

import (
	"github.com/RoaringBitmap/roaring"
)

// ═══════════════════════════════════════════════════════════════════════════════
// QUERY BUILDER: Type-Safe Boolean Queries with Roaring Bitmaps
// ═══════════════════════════════════════════════════════════════════════════════
// Instead of parsing strings like "machine AND learning", use a fluent API:
//
// EXAMPLE USAGE:
// --------------
// Query: Find documents with "machine" AND "learning"
//
//	results := NewQueryBuilder(index).
//	    Term("machine").
//	    And().
//	    Term("learning").
//	    Execute()
//
// Query: Find documents with ("cat" OR "dog") but NOT "snake"
//
//	results := NewQueryBuilder(index).
//	    Group(func(q *QueryBuilder) {
//	        q.Term("cat").Or().Term("dog")
//	    }).
//	    And().Not().Term("snake").
//	    Execute()
//
// WHY BUILDER PATTERN?
// --------------------
// ✓ Type-safe: Compiler catches errors
// ✓ IDE-friendly: Auto-completion works
// ✓ Fluent: Reads like natural language
// ✓ Fast: Direct bitmap operations (no parsing overhead)
// ✓ Composable: Easy to build complex queries programmatically
// ═══════════════════════════════════════════════════════════════════════════════

// QueryBuilder provides a fluent interface for building boolean queries
type QueryBuilder struct {
	index  *InvertedIndex
	stack  []*roaring.Bitmap // Stack of intermediate results
	ops    []QueryOp         // Stack of pending operations
	negate bool              // Whether next term should be negated
	terms  []string          // Track terms for BM25 scoring
}

// QueryOp represents a pending boolean operation
type QueryOp int

const (
	OpNone QueryOp = iota
	OpAnd
	OpOr
)

// NewQueryBuilder creates a new query builder
//
// EXAMPLE:
// --------
//
//	qb := NewQueryBuilder(index)
//	results := qb.Term("machine").And().Term("learning").Execute()
func NewQueryBuilder(index *InvertedIndex) *QueryBuilder {
	return &QueryBuilder{
		index:  index,
		stack:  make([]*roaring.Bitmap, 0),
		ops:    make([]QueryOp, 0),
		negate: false,
		terms:  make([]string, 0),
	}
}

// Term adds a term to the query
//
// WHAT IT DOES:
// -------------
// 1. Gets the roaring bitmap for the term (instant document lookup)
// 2. Applies any pending NOT operation
// 3. Combines with previous results using AND/OR
//
// EXAMPLE:
// --------
//
//	qb.Term("machine")  // Find all docs with "machine"
//
// PERFORMANCE:
// ------------
// O(1) bitmap lookup - no skip list traversal needed!
func (qb *QueryBuilder) Term(term string) *QueryBuilder {
	// Analyze the term (lowercase, stem, etc.)
	tokens := Analyze(term)
	if len(tokens) == 0 {
		// Empty term - push empty bitmap
		qb.pushBitmap(roaring.NewBitmap())
		return qb
	}

	// Track term for BM25 scoring (if not negated)
	analyzedTerm := tokens[0]
	if !qb.negate {
		qb.terms = append(qb.terms, analyzedTerm)
	}

	// Get bitmap for the analyzed term
	bitmap := qb.getTermBitmap(analyzedTerm)

	// Apply negation if needed
	if qb.negate {
		bitmap = qb.negateBitmap(bitmap)
		qb.negate = false
	}

	qb.pushBitmap(bitmap)
	return qb
}

// Phrase adds a phrase query (exact sequence of words)
//
// WHAT IT DOES:
// -------------
// 1. Analyzes the phrase (just like during indexing)
// 2. Uses skip lists to find exact phrase matches
// 3. Converts results to a bitmap for boolean operations
//
// EXAMPLE:
// --------
//
//	qb.Phrase("machine learning")  // Find exact phrase
//
// NOTE: Phrase queries need position information, so we use skip lists
func (qb *QueryBuilder) Phrase(phrase string) *QueryBuilder {
	// Analyze the phrase to match what was indexed
	// This converts "Machine Learning" to "machin learn" etc.
	tokens := Analyze(phrase)
	if len(tokens) == 0 {
		qb.pushBitmap(roaring.NewBitmap())
		return qb
	}

	// Track terms for BM25 scoring (if not negated)
	if !qb.negate {
		qb.terms = append(qb.terms, tokens...)
	}

	// Reconstruct the analyzed phrase
	analyzedPhrase := ""
	for i, token := range tokens {
		if i > 0 {
			analyzedPhrase += " "
		}
		analyzedPhrase += token
	}

	// Use existing phrase search from skip lists
	matches := qb.index.FindAllPhrases(analyzedPhrase, BOFDocument)

	// Convert to bitmap
	bitmap := roaring.NewBitmap()
	for _, match := range matches {
		if !match[0].IsEnd() {
			bitmap.Add(uint32(match[0].GetDocumentID()))
		}
	}

	// Apply negation if needed
	if qb.negate {
		bitmap = qb.negateBitmap(bitmap)
		qb.negate = false
	}

	qb.pushBitmap(bitmap)
	return qb
}

// And adds an AND operation
//
// EXAMPLE:
// --------
//
//	qb.Term("machine").And().Term("learning")
//	// Returns docs with BOTH "machine" AND "learning"
//
// PERFORMANCE:
// ------------
// Roaring bitmap intersection: O(1) for compressed chunks
func (qb *QueryBuilder) And() *QueryBuilder {
	qb.ops = append(qb.ops, OpAnd)
	return qb
}

// Or adds an OR operation
//
// EXAMPLE:
// --------
//
//	qb.Term("cat").Or().Term("dog")
//	// Returns docs with "cat" OR "dog" (or both)
//
// PERFORMANCE:
// ------------
// Roaring bitmap union: O(1) for compressed chunks
func (qb *QueryBuilder) Or() *QueryBuilder {
	qb.ops = append(qb.ops, OpOr)
	return qb
}

// Not negates the next term
//
// EXAMPLE:
// --------
//
//	qb.Term("python").And().Not().Term("snake")
//	// Returns docs with "python" but NOT "snake"
//
// PERFORMANCE:
// ------------
// Roaring bitmap difference: O(1) for compressed chunks
func (qb *QueryBuilder) Not() *QueryBuilder {
	qb.negate = true
	return qb
}

// Group creates a sub-query with its own scope
//
// EXAMPLE:
// --------
//
//	qb.Group(func(q *QueryBuilder) {
//	    q.Term("cat").Or().Term("dog")
//	}).And().Term("pet")
//	// Returns: (cat OR dog) AND pet
//
// USE CASE: Control operator precedence
func (qb *QueryBuilder) Group(fn func(*QueryBuilder)) *QueryBuilder {
	// Create a new sub-query
	subQuery := NewQueryBuilder(qb.index)

	// Execute the group function
	fn(subQuery)

	// Get the result from the sub-query
	result := subQuery.Execute()

	// Apply negation if needed
	if qb.negate {
		result = qb.negateBitmap(result)
		qb.negate = false
	}

	qb.pushBitmap(result)
	return qb
}

// Execute runs the query and returns matching document IDs as a bitmap
//
// ALGORITHM:
// ----------
// 1. Process all terms and operations in order
// 2. Apply AND/OR operations using roaring bitmap operations
// 3. Return final bitmap of matching documents
//
// EXAMPLE:
// --------
//
//	qb := NewQueryBuilder(index)
//	results := qb.Term("machine").And().Term("learning").Execute()
//	// results is a roaring.Bitmap with doc IDs
//
// PERFORMANCE:
// ------------
// All operations use optimized roaring bitmap operations:
// - AND: bitmap intersection (fast!)
// - OR: bitmap union (fast!)
// - NOT: bitmap difference (fast!)
func (qb *QueryBuilder) Execute() *roaring.Bitmap {
	if len(qb.stack) == 0 {
		return roaring.NewBitmap()
	}

	// Process the stack with operations
	result := qb.stack[0]
	for i := 1; i < len(qb.stack); i++ {
		if i-1 < len(qb.ops) {
			op := qb.ops[i-1]
			switch op {
			case OpAnd:
				// Intersection: docs in BOTH bitmaps
				result = roaring.And(result, qb.stack[i])
			case OpOr:
				// Union: docs in EITHER bitmap
				result = roaring.Or(result, qb.stack[i])
			}
		}
	}

	return result
}

// ExecuteWithBM25 runs the query and returns ranked results using BM25
//
// ALGORITHM:
// ----------
// 1. Execute boolean query → Get bitmap of matching docs
// 2. Extract terms from the query
// 3. Calculate BM25 score for each matching document
// 4. Sort by score and return top K
//
// EXAMPLE:
// --------
//
//	qb := NewQueryBuilder(index)
//	matches := qb.Term("machine").And().Term("learning").
//	    ExecuteWithBM25(10)
//	// Returns top 10 matches sorted by BM25 score
func (qb *QueryBuilder) ExecuteWithBM25(maxResults int) []Match {
	// Execute boolean query
	resultBitmap := qb.Execute()

	// Extract terms for BM25 scoring
	terms := qb.extractTerms()

	// Score each matching document
	var results []Match
	iter := resultBitmap.Iterator()
	for iter.HasNext() {
		docID := int(iter.Next())
		score := qb.index.calculateBM25Score(docID, terms)

		if score > 0 {
			results = append(results, Match{
				DocID: docID,
				Score: score,
			})
		}
	}

	// Sort by score (descending)
	qb.index.sortMatchesByScore(results)

	// Return top K
	return limitResults(results, maxResults)
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNAL HELPER METHODS
// ═══════════════════════════════════════════════════════════════════════════════

// getTermBitmap retrieves the roaring bitmap for a term
func (qb *QueryBuilder) getTermBitmap(term string) *roaring.Bitmap {
	if bitmap, exists := qb.index.DocBitmaps[term]; exists {
		return bitmap.Clone() // Clone to avoid modifying original
	}
	return roaring.NewBitmap() // Empty bitmap if term not found
}

// negateBitmap returns all documents EXCEPT those in the bitmap
func (qb *QueryBuilder) negateBitmap(bitmap *roaring.Bitmap) *roaring.Bitmap {
	// Create bitmap of all documents
	allDocs := roaring.NewBitmap()
	for docID := range qb.index.DocStats {
		allDocs.Add(uint32(docID))
	}

	// Return difference: all docs - bitmap
	return roaring.AndNot(allDocs, bitmap)
}

// pushBitmap pushes a bitmap onto the stack
func (qb *QueryBuilder) pushBitmap(bitmap *roaring.Bitmap) {
	qb.stack = append(qb.stack, bitmap)
}

// extractTerms extracts all terms used in the query for BM25 scoring
func (qb *QueryBuilder) extractTerms() []string {
	return qb.terms
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVENIENCE METHODS FOR COMMON PATTERNS
// ═══════════════════════════════════════════════════════════════════════════════

// AllOf finds documents containing ALL of the given terms (AND)
//
// EXAMPLE:
// --------
//
//	results := AllOf(index, "machine", "learning", "python")
//	// Same as: Term("machine").And().Term("learning").And().Term("python")
func AllOf(index *InvertedIndex, terms ...string) *roaring.Bitmap {
	if len(terms) == 0 {
		return roaring.NewBitmap()
	}

	qb := NewQueryBuilder(index).Term(terms[0])
	for i := 1; i < len(terms); i++ {
		qb.And().Term(terms[i])
	}
	return qb.Execute()
}

// AnyOf finds documents containing ANY of the given terms (OR)
//
// EXAMPLE:
// --------
//
//	results := AnyOf(index, "cat", "dog", "bird")
//	// Same as: Term("cat").Or().Term("dog").Or().Term("bird")
func AnyOf(index *InvertedIndex, terms ...string) *roaring.Bitmap {
	if len(terms) == 0 {
		return roaring.NewBitmap()
	}

	qb := NewQueryBuilder(index).Term(terms[0])
	for i := 1; i < len(terms); i++ {
		qb.Or().Term(terms[i])
	}
	return qb.Execute()
}

// TermExcluding finds documents with a term but excluding another
//
// EXAMPLE:
// --------
//
//	results := TermExcluding(index, "python", "snake")
//	// Same as: Term("python").And().Not().Term("snake")
func TermExcluding(index *InvertedIndex, include, exclude string) *roaring.Bitmap {
	return NewQueryBuilder(index).
		Term(include).
		And().Not().Term(exclude).
		Execute()
}
