package iskra

import (
	"github.com/RoaringBitmap/roaring"
)

// ═══════════════════════════════════════════════════════════════════════════════
// QUERY BUILDER: Type-Safe Boolean Queries with Roaring Bitmaps
// ═══════════════════════════════════════════════════════════════════════════════

// QueryBuilder provides a fluent interface for building boolean queries
type QueryBuilder struct {
	index  *InvertedIndex
	stack  []*roaring.Bitmap
	ops    []QueryOp
	negate bool
	terms  []string // Track terms for BM25 scoring
}

// QueryOp represents a pending boolean operation
type QueryOp int

const (
	OpNone QueryOp = iota
	OpAnd
	OpOr
)

// NewQueryBuilder creates a new query builder
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
func (qb *QueryBuilder) Term(term string) *QueryBuilder {
	tokens := Analyze(term)
	if len(tokens) == 0 {
		qb.pushBitmap(roaring.NewBitmap())
		return qb
	}

	analyzedTerm := tokens[0]
	if !qb.negate {
		qb.terms = append(qb.terms, analyzedTerm)
	}

	bitmap := qb.getTermBitmap(analyzedTerm)

	if qb.negate {
		bitmap = qb.negateBitmap(bitmap)
		qb.negate = false
	}

	qb.pushBitmap(bitmap)
	return qb
}

// Phrase adds a phrase query (exact sequence of words)
func (qb *QueryBuilder) Phrase(phrase string) *QueryBuilder {
	tokens := Analyze(phrase)
	if len(tokens) == 0 {
		qb.pushBitmap(roaring.NewBitmap())
		return qb
	}

	if !qb.negate {
		qb.terms = append(qb.terms, tokens...)
	}

	analyzedPhrase := ""
	for i, token := range tokens {
		if i > 0 {
			analyzedPhrase += " "
		}
		analyzedPhrase += token
	}

	matches := qb.index.FindAllPhrases(analyzedPhrase, BOFDocument)

	bitmap := roaring.NewBitmap()
	for _, match := range matches {
		if !match[0].IsEnd() {
			bitmap.Add(match[0].GetDocumentID())
		}
	}

	if qb.negate {
		bitmap = qb.negateBitmap(bitmap)
		qb.negate = false
	}

	qb.pushBitmap(bitmap)
	return qb
}

// And adds an AND operation
func (qb *QueryBuilder) And() *QueryBuilder {
	qb.ops = append(qb.ops, OpAnd)
	return qb
}

// Or adds an OR operation
func (qb *QueryBuilder) Or() *QueryBuilder {
	qb.ops = append(qb.ops, OpOr)
	return qb
}

// Not negates the next term
func (qb *QueryBuilder) Not() *QueryBuilder {
	qb.negate = true
	return qb
}

// Group creates a sub-query with its own scope
func (qb *QueryBuilder) Group(fn func(*QueryBuilder)) *QueryBuilder {
	subQuery := NewQueryBuilder(qb.index)
	fn(subQuery)
	result := subQuery.Execute()

	if qb.negate {
		result = qb.negateBitmap(result)
		qb.negate = false
	}

	qb.pushBitmap(result)
	return qb
}

// Execute runs the query and returns matching document IDs as a bitmap
func (qb *QueryBuilder) Execute() *roaring.Bitmap {
	if len(qb.stack) == 0 {
		return roaring.NewBitmap()
	}

	result := qb.stack[0]
	for i := 1; i < len(qb.stack); i++ {
		if i-1 < len(qb.ops) {
			switch qb.ops[i-1] {
			case OpAnd:
				result = roaring.And(result, qb.stack[i])
			case OpOr:
				result = roaring.Or(result, qb.stack[i])
			}
		}
	}
	return result
}

// ExecuteWithBM25 runs the query and returns ranked results using BM25
func (qb *QueryBuilder) ExecuteWithBM25(maxResults int) []Match {
	resultBitmap := qb.Execute()
	terms := qb.extractTerms()

	var results []Match
	iter := resultBitmap.Iterator()
	for iter.HasNext() {
		docID := iter.Next() // uint32 natively from roaring
		score := qb.index.calculateBM25Score(docID, terms)
		if score > 0 {
			results = append(results, Match{
				DocID: docID,
				Score: score,
			})
		}
	}

	qb.index.sortMatchesByScore(results)
	return limitResults(results, maxResults)
}

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNAL HELPER METHODS
// ═══════════════════════════════════════════════════════════════════════════════

// getTermBitmap retrieves the roaring bitmap for a term
func (qb *QueryBuilder) getTermBitmap(term string) *roaring.Bitmap {
	if bitmap, exists := qb.index.DocBitmaps[term]; exists {
		return bitmap.Clone()
	}
	return roaring.NewBitmap()
}

// negateBitmap returns all documents EXCEPT those in the bitmap
func (qb *QueryBuilder) negateBitmap(bitmap *roaring.Bitmap) *roaring.Bitmap {
	allDocs := roaring.NewBitmap()
	for docID := range qb.index.DocStats {
		allDocs.Add(docID) // docID is already uint32
	}
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
func TermExcluding(index *InvertedIndex, include, exclude string) *roaring.Bitmap {
	return NewQueryBuilder(index).
		Term(include).
		And().Not().Term(exclude).
		Execute()
}
