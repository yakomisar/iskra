package iskra

import (
	"errors"
	"math"
	"math/rand"
	"time"
)

// ═══════════════════════════════════════════════════════════════════════════════
// WHAT IS A SKIP LIST?
// ═══════════════════════════════════════════════════════════════════════════════
// A skip list is a probabilistic data structure that allows O(log n) search,
// insert, and delete operations - similar to a balanced tree, but simpler!
//
// VISUAL REPRESENTATION:
// ----------------------
// Think of it as a linked list with "express lanes":
//
// Level 3: HEAD -------------------------------------> [30] -----------> NULL
// Level 2: HEAD ----------------> [15] -------------> [30] -----------> NULL
// Level 1: HEAD -------> [10] --> [15] --> [20] ----> [30] -----------> NULL
// Level 0: HEAD --> [5] -> [10] -> [15] -> [20] -> [25] -> [30] -> [35] -> NULL
//                   ^^^    ^^^     ^^^     ^^^     ^^^     ^^^     ^^^
//                  Actual  data    in      the     skip    list    nodes
//
// HOW IT WORKS:
// -------------
// - Level 0 (bottom): Contains ALL elements in sorted order
// - Higher levels: Contain progressively fewer elements (like express lanes)
// - Searching: Start at the highest level, drop down when needed
//
// SEARCH EXAMPLE (finding 20):
// -----------------------------
// 1. Start at HEAD, Level 3
// 2. Level 3: Move to 30? No, 30 > 20, so drop to Level 2
// 3. Level 2: Move to 15? Yes, 15 < 20, advance to 15
// 4. Level 2: Move to 30? No, 30 > 20, so drop to Level 1
// 5. Level 1: Move to 20? Yes! Found it!
//
// Time Complexity: O(log n) average case
// - Each level skips roughly half the elements
// - Similar to binary search, but on a linked structure
//
// WHY USE SKIP LISTS IN A SEARCH ENGINE?
// ---------------------------------------
// 1. Fast lookups: O(log n) to find any position
// 2. Fast range queries: Find all positions in a document efficiently
// 3. Maintains sorted order: Essential for phrase search
// 4. Simple implementation: Easier than balanced trees (no rotations!)
// 5. Good cache locality: Level 0 can be traversed sequentially
//
// ═══════════════════════════════════════════════════════════════════════════════

const MaxHeight = 32 // Maximum tower height (supports billions of elements)

// ═══════════════════════════════════════════════════════════════════════════════
// SENTINEL VALUES
// ═══════════════════════════════════════════════════════════════════════════════
// We use MaxUint32 and 0 (with a flag) as boundary markers.
// Since DocID is now uint32, we use a separate sentinel type.
//
// WHY KEEP int FOR Offset?
// -------------------------
// Offsets are word positions within a document — they never need to be
// document IDs, so they stay as int with MinInt/MaxInt sentinels.
var (
	EOF = math.MaxInt // End Of File sentinel for Offset
	BOF = math.MinInt // Beginning Of File sentinel for Offset
)

var (
	ErrKeyNotFound    = errors.New("key not found")
	ErrNoElementFound = errors.New("no element found")
)

// ═══════════════════════════════════════════════════════════════════════════════
// POSITION: A Location in a Document
// ═══════════════════════════════════════════════════════════════════════════════
// Position identifies a specific word in a specific document.
//
// DocumentID is uint32: non-negative, matches roaring.Bitmap natively,
// and covers ~4 billion documents — more than enough for any use case.
//
// Offset stays int so it can hold BOF (MinInt) and EOF (MaxInt) sentinels.
//
// ORDERING:
// ---------
// Positions are ordered first by DocumentID, then by Offset:
//
//	Doc1:Pos5 < Doc1:Pos10 < Doc2:Pos0 < Doc2:Pos3
//
// ═══════════════════════════════════════════════════════════════════════════════
type Position struct {
	DocumentID uint32 // Which document? (uint32 matches roaring.Bitmap natively)
	Offset     int    // Which word in the document? (int to support BOF/EOF sentinels)
}

// Sentinel positions for convenience
var (
	BOFDocument = Position{DocumentID: 0, Offset: BOF}              // Before all documents
	EOFDocument = Position{DocumentID: math.MaxUint32, Offset: EOF} // After all documents
)

// ═══════════════════════════════════════════════════════════════════════════════
// POSITION HELPER METHODS
// ═══════════════════════════════════════════════════════════════════════════════

// GetDocumentID returns the document ID
func (p *Position) GetDocumentID() uint32 {
	return p.DocumentID
}

// GetOffset returns the offset
func (p *Position) GetOffset() int {
	return p.Offset
}

// IsBeginning checks if this is the BOF sentinel
func (p *Position) IsBeginning() bool {
	return p.Offset == BOF
}

// IsEnd checks if this is the EOF sentinel
func (p *Position) IsEnd() bool {
	return p.Offset == EOF
}

// IsBefore checks if this position comes before another position
//
// ORDERING RULES:
// ---------------
// Position A < Position B if:
//  1. A.DocumentID < B.DocumentID, OR
//  2. Same document AND A.Offset < B.Offset
func (p *Position) IsBefore(other Position) bool {
	if p.DocumentID < other.DocumentID {
		return true
	}
	return p.DocumentID == other.DocumentID && p.Offset < other.Offset
}

// IsAfter checks if this position comes after another position
func (p *Position) IsAfter(other Position) bool {
	if p.DocumentID > other.DocumentID {
		return true
	}
	return p.DocumentID == other.DocumentID && p.Offset > other.Offset
}

// Equals checks if two positions are identical
func (p *Position) Equals(other Position) bool {
	return p.DocumentID == other.DocumentID && p.Offset == other.Offset
}

// ═══════════════════════════════════════════════════════════════════════════════
// NODE: A Skip List Node
// ═══════════════════════════════════════════════════════════════════════════════
type Node struct {
	Key   Position         // The position stored in this node
	Tower [MaxHeight]*Node // Array of forward pointers (one per level)
}

// ═══════════════════════════════════════════════════════════════════════════════
// SKIP LIST: The Main Data Structure
// ═══════════════════════════════════════════════════════════════════════════════
type SkipList struct {
	Head   *Node // Sentinel head node (doesn't contain real data)
	Height int   // Current height of the tallest tower
}

// NewSkipList creates an empty skip list
func NewSkipList() *SkipList {
	return &SkipList{
		Head:   &Node{}, // Empty sentinel head
		Height: 1,
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEARCH: The Core Operation
// ═══════════════════════════════════════════════════════════════════════════════

// Search finds a key in the skip list and returns the path taken
func (sl *SkipList) Search(key Position) (*Node, [MaxHeight]*Node) {
	var journey [MaxHeight]*Node
	current := sl.Head

	for level := sl.Height - 1; level >= 0; level-- {
		current = sl.traverseLevel(current, key, level)
		journey[level] = current
	}

	next := current.Tower[0]
	if next != nil && next.Key.Equals(key) {
		return next, journey
	}
	return nil, journey
}

// traverseLevel advances along a single level as far as possible
func (sl *SkipList) traverseLevel(start *Node, target Position, level int) *Node {
	current := start
	next := current.Tower[level]
	for next != nil {
		if sl.shouldAdvance(next.Key, target) {
			current = next
			next = current.Tower[level]
		} else {
			break
		}
	}
	return current
}

// shouldAdvance determines if we should move to the next node
func (sl *SkipList) shouldAdvance(nodeKey, targetKey Position) bool {
	if nodeKey.Equals(targetKey) {
		return false
	}
	return nodeKey.IsBefore(targetKey)
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIND OPERATIONS: Building on Search
// ═══════════════════════════════════════════════════════════════════════════════

// Find searches for an exact key match
func (sl *SkipList) Find(key Position) (Position, error) {
	found, _ := sl.Search(key)
	if found == nil {
		return EOFDocument, ErrKeyNotFound
	}
	return found.Key, nil
}

// FindLessThan finds the largest key less than the given key
func (sl *SkipList) FindLessThan(key Position) (Position, error) {
	_, journey := sl.Search(key)
	predecessor := journey[0]

	if predecessor == nil || predecessor == sl.Head {
		return BOFDocument, ErrNoElementFound
	}
	return predecessor.Key, nil
}

// FindGreaterThan finds the smallest key greater than the given key
func (sl *SkipList) FindGreaterThan(key Position) (Position, error) {
	found, journey := sl.Search(key)

	if found != nil {
		if found.Tower[0] != nil {
			return found.Tower[0].Key, nil
		}
		return EOFDocument, ErrNoElementFound
	}

	predecessor := journey[0]
	if predecessor != nil && predecessor.Tower[0] != nil {
		return predecessor.Tower[0].Key, nil
	}
	return EOFDocument, ErrNoElementFound
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSERT: Adding Elements to the Skip List
// ═══════════════════════════════════════════════════════════════════════════════

// Insert adds a new key to the skip list (or updates if it exists)
func (sl *SkipList) Insert(key Position) {
	found, journey := sl.Search(key)

	if found != nil {
		found.Key = key
		return
	}

	height := sl.randomHeight()
	newNode := &Node{Key: key}
	sl.linkNode(newNode, journey, height)

	if height > sl.Height {
		sl.Height = height
	}
}

// linkNode connects a new node into the skip list structure
func (sl *SkipList) linkNode(node *Node, journey [MaxHeight]*Node, height int) {
	for level := 0; level < height; level++ {
		predecessor := journey[level]
		if predecessor == nil {
			predecessor = sl.Head
		}
		node.Tower[level] = predecessor.Tower[level]
		predecessor.Tower[level] = node
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE: Removing Elements from the Skip List
// ═══════════════════════════════════════════════════════════════════════════════

// Delete removes a key from the skip list
func (sl *SkipList) Delete(key Position) bool {
	found, journey := sl.Search(key)

	if found == nil {
		return false
	}

	for level := 0; level < sl.Height; level++ {
		if journey[level].Tower[level] != found {
			break
		}
		journey[level].Tower[level] = found.Tower[level]
	}

	sl.shrink()
	return true
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Last returns the last position in the skip list
func (sl *SkipList) Last() Position {
	current := sl.Head
	for next := current.Tower[0]; next != nil; next = next.Tower[0] {
		current = next
	}
	return current.Key
}

// shrink reduces the height if top levels are empty
func (sl *SkipList) shrink() {
	for level := sl.Height - 1; level >= 0; level-- {
		if sl.Head.Tower[level] == nil {
			sl.Height--
		} else {
			break
		}
	}
}

// randomHeight generates a random height for a new node
func (sl *SkipList) randomHeight() int {
	height := 1
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for rng.Float64() < 0.5 && height < MaxHeight {
		height++
	}
	return height
}

// ═══════════════════════════════════════════════════════════════════════════════
// ITERATOR: Sequential Access to Elements
// ═══════════════════════════════════════════════════════════════════════════════

// Iterator provides sequential access to skip list elements
type Iterator struct {
	current *Node
}

// Iterator creates a new iterator starting at the first element
func (sl *SkipList) Iterator() *Iterator {
	return &Iterator{current: sl.Head.Tower[0]}
}

// HasNext checks if there are more elements to iterate
func (it *Iterator) HasNext() bool {
	return it.current != nil && it.current.Tower[0] != nil
}

// Next advances to and returns the next position
func (it *Iterator) Next() Position {
	if it.current == nil {
		return EOFDocument
	}
	it.current = it.current.Tower[0]
	if it.current == nil {
		return EOFDocument
	}
	return it.current.Key
}
