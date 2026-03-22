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
// We use MaxInt and MinInt as boundary markers
//
// WHY USE MAX/MIN INT?
// --------------------
// - Makes comparisons cleaner (no special cases for "empty")
// - Always guarantees: BOF < any_position < EOF
// - Simplifies edge cases in search algorithms
//
// Example: Searching from the "beginning"
//
//	Without sentinels: Need to check "is this the first call?"
//	With sentinels: Just use BOF as the starting position!
var (
	EOF = math.MaxInt // End Of File: maximum integer value (larger than any real position)
	BOF = math.MinInt // Beginning Of File: minimum integer value (smaller than any real position)
)

var (
	ErrKeyNotFound    = errors.New("key not found")
	ErrNoElementFound = errors.New("no element found")
)

// ═══════════════════════════════════════════════════════════════════════════════
// POSITION: A Location in a Document
// ═══════════════════════════════════════════════════════════════════════════════
// Position identifies a specific word in a specific document
//
// EXAMPLE:
// --------
// Document 5: "The quick brown fox jumps"
// Position{DocumentID: 5, Offset: 2} refers to "brown"
//
// WHY USE INT?
// ------------
// - Represents actual integer document IDs and offsets
// - Supports sentinel values (BOF = MinInt, EOF = MaxInt)
// - No casting needed - simpler and more efficient
//
// ORDERING:
// ---------
// Positions are ordered first by DocumentID, then by Offset:
//
//	Doc1:Pos5 < Doc1:Pos10 < Doc2:Pos0 < Doc2:Pos3
//
// ═══════════════════════════════════════════════════════════════════════════════
type Position struct {
	DocumentID int // Which document?
	Offset     int // Which word in the document? (0-indexed)
}

// Sentinel positions for convenience
var (
	BOFDocument = Position{DocumentID: BOF, Offset: BOF} // Before all documents
	EOFDocument = Position{DocumentID: EOF, Offset: EOF} // After all documents
)

// ═══════════════════════════════════════════════════════════════════════════════
// POSITION HELPER METHODS
// ═══════════════════════════════════════════════════════════════════════════════
// These methods make Position comparisons more readable and less error-prone
// ═══════════════════════════════════════════════════════════════════════════════

// GetDocumentID returns the document ID
// (Convenience method for consistent API)
func (p *Position) GetDocumentID() int {
	return p.DocumentID
}

// GetOffset returns the offset
// (Convenience method for consistent API)
func (p *Position) GetOffset() int {
	return p.Offset
}

// IsBeginning checks if this is the BOF sentinel
//
// Example usage:
//
//	if pos.IsBeginning() {
//	    // We're at the start, no previous element exists
//	}
func (p *Position) IsBeginning() bool {
	return p.Offset == BOF
}

// IsEnd checks if this is the EOF sentinel
//
// Example usage:
//
//	if pos.IsEnd() {
//	    // We've reached the end, stop searching
//	}
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
//
// EXAMPLES:
// ---------
// Doc1:Pos5 < Doc1:Pos10 → true  (same doc, 5 < 10)
// Doc1:Pos5 < Doc2:Pos0  → true  (doc 1 < doc 2)
// Doc2:Pos0 < Doc1:Pos5  → false (doc 2 > doc 1)
func (p *Position) IsBefore(other Position) bool {
	// Check document order first
	if p.DocumentID < other.DocumentID {
		return true
	}

	// Same document: check offset order
	return p.DocumentID == other.DocumentID && p.Offset < other.Offset
}

// IsAfter checks if this position comes after another position
//
// This is the opposite of IsBefore (with equality handled separately)
func (p *Position) IsAfter(other Position) bool {
	// Check document order first
	if p.DocumentID > other.DocumentID {
		return true
	}

	// Same document: check offset order
	return p.DocumentID == other.DocumentID && p.Offset > other.Offset
}

// Equals checks if two positions are identical
//
// Example:
//
//	Doc1:Pos5 == Doc1:Pos5 → true
//	Doc1:Pos5 == Doc1:Pos6 → false
func (p *Position) Equals(other Position) bool {
	return p.DocumentID == other.DocumentID && p.Offset == other.Offset
}

// ═══════════════════════════════════════════════════════════════════════════════
// NODE: A Skip List Node
// ═══════════════════════════════════════════════════════════════════════════════
// Each node stores:
// 1. A Key (Position): The data we're storing
// 2. A Tower: Array of pointers to next nodes at each level
//
// TOWER VISUALIZATION:
// --------------------
// For a node with height 3:
//
//	Tower[2] -----> (points to a node far ahead)
//	Tower[1] -----> (points to a node ahead)
//	Tower[0] -----> (points to the very next node)
//
// The higher the level, the further ahead we skip!
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
//
// INITIAL STATE:
// --------------
// HEAD (empty node) with no forward pointers
// Height = 1 (even empty lists have level 0)
func NewSkipList() *SkipList {
	return &SkipList{
		Head:   &Node{}, // Empty sentinel head
		Height: 1,
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEARCH: The Core Operation
// ═══════════════════════════════════════════════════════════════════════════════
// Search is the foundation of all skip list operations.
// It returns TWO things:
// 1. The node with the exact key (or nil if not found)
// 2. A "journey" array: the path we took to get there
//
// WHY RETURN THE JOURNEY?
// ------------------------
// The journey tells us which node is BEFORE the target at each level.
// This is essential for:
// - Insert: We need to know where to splice in the new node
// - Delete: We need to know which nodes to update
// - FindLessThan: The journey already contains the answer!
//
// SEARCH ALGORITHM:
// -----------------
// Start at the highest level and work down:
// 1. At each level, move right as far as possible (while staying < target)
// 2. When we can't move right, drop down one level
// 3. Repeat until we reach level 0
// 4. Check if we found the exact key
//
// VISUAL EXAMPLE (searching for 20):
// -----------------------------------
// Level 2: HEAD ------[10]------[30]     Start at HEAD, level 2
//                     ^^^                Can we jump to 10? Yes! (10 < 20)
//                           ^^^          Can we jump to 30? No! (30 > 20)
//                                        Drop to level 1...
//
// Level 1: HEAD --[10]--[15]--[20]--[30]  At 10, level 1
//                       ^^^                Can we jump to 15? Yes! (15 < 20)
//                             ^^^          Can we jump to 20? STOP! Check this
//
// Level 0: We'd check if 20 exists at level 0
//
// Journey captured: [level0: node15, level1: node15, level2: node10]
// ═══════════════════════════════════════════════════════════════════════════════

// Search finds a key in the skip list and returns the path taken
//
// RETURN VALUES:
// --------------
// 1. *Node: The node with exact key (nil if not found)
// 2. [MaxHeight]*Node: Journey array - the predecessor at each level
//
// EXAMPLE:
// --------
// Skip list: [5] -> [10] -> [15] -> [20]
// Search(15) returns:
//   - found: Node{15}
//   - journey[0]: Node{10} (predecessor at level 0)
//   - journey[1]: Node{10} (predecessor at level 1)
//   - ...
func (sl *SkipList) Search(key Position) (*Node, [MaxHeight]*Node) {
	var journey [MaxHeight]*Node // Track the path we take
	current := sl.Head           // Start at the sentinel head

	// Traverse from highest level down to level 0
	for level := sl.Height - 1; level >= 0; level-- {
		// Move forward as far as possible at this level
		current = sl.traverseLevel(current, key, level)

		// Record where we ended up at this level
		// (This is the predecessor for this level)
		journey[level] = current
	}

	// Check if we found an exact match
	// current now points to the largest node < key
	// So current.Tower[0] might be the exact key
	next := current.Tower[0]
	if next != nil && next.Key.Equals(key) {
		return next, journey // Found it!
	}

	return nil, journey // Not found, but journey is still useful
}

// traverseLevel advances along a single level as far as possible
//
// PROCESS:
// --------
// Starting from 'start', move forward while next.Key < target
// Stop when: next.Key >= target OR next == nil
//
// EXAMPLE:
// --------
// Level: HEAD -> [5] -> [10] -> [15] -> [20] -> nil
// Target: 17
//
// Step 1: At HEAD, next = 5, should advance? Yes (5 < 17)
// Step 2: At 5, next = 10, should advance? Yes (10 < 17)
// Step 3: At 10, next = 15, should advance? Yes (15 < 17)
// Step 4: At 15, next = 20, should advance? No! (20 > 17)
// Return: node 15
func (sl *SkipList) traverseLevel(start *Node, target Position, level int) *Node {
	current := start

	// Keep moving forward while we can
	next := current.Tower[level]
	for next != nil {
		// Should we advance to the next node?
		if sl.shouldAdvance(next.Key, target) {
			current = next              // Yes, move forward
			next = current.Tower[level] // Update next to the next node
		} else {
			break // No, stop here
		}
	}

	return current
}

// shouldAdvance determines if we should move to the next node
//
// DECISION RULE:
// --------------
// Advance if: next.Key < target
// Stop if: next.Key >= target
//
// This ensures we stop at the largest node that's still less than target
func (sl *SkipList) shouldAdvance(nodeKey, targetKey Position) bool {
	// Don't advance if we've reached or passed the target
	if nodeKey.Equals(targetKey) {
		return false
	}

	// Advance only if the node key is less than target
	return nodeKey.IsBefore(targetKey)
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIND OPERATIONS: Building on Search
// ═══════════════════════════════════════════════════════════════════════════════
// These operations use Search as a building block
// ═══════════════════════════════════════════════════════════════════════════════

// Find searches for an exact key match
//
// # This is a simple wrapper around Search that only returns the key
//
// Example:
//
//	Find(Doc1:Pos5) returns Doc1:Pos5 if it exists, else error
func (sl *SkipList) Find(key Position) (Position, error) {
	found, _ := sl.Search(key)

	if found == nil {
		return EOFDocument, ErrKeyNotFound
	}

	return found.Key, nil
}

// FindLessThan finds the largest key less than the given key
//
// HOW IT WORKS:
// -------------
// The journey from Search already gives us this answer!
// journey[0] is the largest node < key at the bottom level
//
// EXAMPLE:
// --------
// Skip list: [5] -> [10] -> [15] -> [20]
// FindLessThan(17) returns 15
// FindLessThan(15) returns 10
// FindLessThan(5) returns BOF (nothing before 5)
//
// USE CASE:
// ---------
// In search: "Find the previous occurrence of 'quick' before position X"
func (sl *SkipList) FindLessThan(key Position) (Position, error) {
	_, journey := sl.Search(key)

	predecessor := journey[0] // The node before key at level 0

	// Check edge cases
	if predecessor == nil || predecessor == sl.Head {
		return BOFDocument, ErrNoElementFound
	}

	return predecessor.Key, nil
}

// FindGreaterThan finds the smallest key greater than the given key
//
// TWO CASES:
// ----------
// 1. Key exists: Return the next node after it
// 2. Key doesn't exist: Return the next node after where it would be
//
// EXAMPLE:
// --------
// Skip list: [5] -> [10] -> [15] -> [20]
// FindGreaterThan(10) returns 15 (next after 10)
// FindGreaterThan(12) returns 15 (next after where 12 would be)
// FindGreaterThan(20) returns EOF (nothing after 20)
//
// USE CASE:
// ---------
// In search: "Find the next occurrence of 'quick' after position X"
func (sl *SkipList) FindGreaterThan(key Position) (Position, error) {
	found, journey := sl.Search(key)

	// CASE 1: Key exists - return its successor
	if found != nil {
		if found.Tower[0] != nil {
			return found.Tower[0].Key, nil
		}
		return EOFDocument, ErrNoElementFound
	}

	// CASE 2: Key doesn't exist - return next node after where it would be
	predecessor := journey[0]
	if predecessor != nil && predecessor.Tower[0] != nil {
		return predecessor.Tower[0].Key, nil
	}

	return EOFDocument, ErrNoElementFound
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSERT: Adding Elements to the Skip List
// ═══════════════════════════════════════════════════════════════════════════════
// Insertion is a two-phase process:
// 1. Search to find where the new element should go
// 2. Splice the new node into the list at multiple levels
//
// PROBABILISTIC HEIGHT:
// ---------------------
// Each new node gets a random height (tower height):
// - 50% chance of height 1
// - 25% chance of height 2
// - 12.5% chance of height 3
// - ...
//
// This randomness is what makes skip lists work!
// It ensures roughly logarithmic performance on average.
//
// INSERT EXAMPLE:
// ---------------
// Inserting 17 with height 2:
//
// Before:
// Level 1: HEAD -------> [10] ------------> [20]
// Level 0: HEAD -> [5] -> [10] -> [15] -> [20]
//
// After:
// Level 1: HEAD -------> [10] -> [17] ----> [20]
// Level 0: HEAD -> [5] -> [10] -> [15] -> [17] -> [20]
//                                           ^^^
//                                          new node
// ═══════════════════════════════════════════════════════════════════════════════

// Insert adds a new key to the skip list (or updates if it exists)
//
// ALGORITHM:
// ----------
//  1. Search for the key (get the journey/path)
//  2. If found, update the existing node
//  3. If not found:
//     a. Generate a random height for the new node
//     b. Create the new node
//     c. Link it into the list at each level
//     d. Update the skip list's height if needed
//
// EXAMPLE WALKTHROUGH:
// --------------------
// Inserting Doc2:Pos5 into skip list: [Doc1:Pos3, Doc2:Pos10]
//
// Step 1: Search(Doc2:Pos5)
//   - Not found
//   - journey[0] = Node{Doc1:Pos3} (predecessor at level 0)
//
// Step 2: Generate height = 2 (random)
//
// Step 3: Create Node{Doc2:Pos5}
//
// Step 4: Link at level 0 and level 1:
//   - Level 0: Doc1:Pos3 -> Doc2:Pos5 -> Doc2:Pos10
//   - Level 1: HEAD -> Doc2:Pos5 -> ...
func (sl *SkipList) Insert(key Position) {
	found, journey := sl.Search(key)

	// If key already exists, just update it
	if found != nil {
		found.Key = key
		return
	}

	// Generate a random height for the new node
	height := sl.randomHeight()

	// Create the new node
	newNode := &Node{Key: key}

	// Link the node into the skip list
	sl.linkNode(newNode, journey, height)

	// Update skip list height if necessary
	if height > sl.Height {
		sl.Height = height
	}
}

// linkNode connects a new node into the skip list structure
//
// LINKING PROCESS (for each level):
// ----------------------------------
// 1. Find the predecessor at this level (from journey)
// 2. Set newNode.Tower[level] = predecessor.Tower[level]
// 3. Set predecessor.Tower[level] = newNode
//
// VISUAL EXAMPLE (linking at level 1):
// -------------------------------------
// Before:
//
//	predecessor -> [oldNext]
//
// After:
//
//	predecessor -> [newNode] -> [oldNext]
//
// The newNode "splices" itself between predecessor and oldNext
func (sl *SkipList) linkNode(node *Node, journey [MaxHeight]*Node, height int) {
	// Link the node at each level up to its height
	for level := 0; level < height; level++ {
		predecessor := journey[level]

		// Edge case: If no predecessor at this level, use HEAD
		if predecessor == nil {
			predecessor = sl.Head
		}

		// Splice the node into the linked list at this level
		// 1. New node points to what predecessor was pointing to
		node.Tower[level] = predecessor.Tower[level]
		// 2. Predecessor now points to new node
		predecessor.Tower[level] = node
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE: Removing Elements from the Skip List
// ═══════════════════════════════════════════════════════════════════════════════
// Deletion is the reverse of insertion:
// 1. Search for the key
// 2. Unlink it from all levels
// 3. Clean up: reduce height if top levels are now empty
// ═══════════════════════════════════════════════════════════════════════════════

// Delete removes a key from the skip list
//
// ALGORITHM:
// ----------
//  1. Search for the key
//  2. If not found, return false
//  3. If found:
//     a. Unlink it from all levels
//     b. Shrink the skip list height if needed
//
// EXAMPLE:
// --------
// Deleting 15:
//
// Before:
// Level 1: HEAD -------> [10] -> [15] ----> [20]
// Level 0: HEAD -> [5] -> [10] -> [15] -> [20]
//
// After:
// Level 1: HEAD -------> [10] ------------> [20]
// Level 0: HEAD -> [5] -> [10] ------------> [20]
//
//	(15 removed)
func (sl *SkipList) Delete(key Position) bool {
	found, journey := sl.Search(key)

	// Key doesn't exist
	if found == nil {
		return false
	}

	// Unlink the node from all levels
	for level := 0; level < sl.Height; level++ {
		// If the predecessor at this level doesn't point to our node,
		// we've finished unlinking (node wasn't tall enough for higher levels)
		if journey[level].Tower[level] != found {
			break
		}

		// Bypass the node: predecessor points to node's successor
		journey[level].Tower[level] = found.Tower[level]
	}

	// Clean up: reduce height if top levels are now empty
	sl.shrink()
	return true
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Last returns the last position in the skip list
//
// HOW IT WORKS:
// -------------
// Simply traverse level 0 until we reach the end
//
// Example:
// Skip list: [5] -> [10] -> [15] -> [20] -> nil
// Last() returns 20
func (sl *SkipList) Last() Position {
	current := sl.Head

	// Traverse the bottom level to the end
	for next := current.Tower[0]; next != nil; next = next.Tower[0] {
		current = next
	}

	return current.Key
}

// shrink reduces the height if top levels are empty
//
// WHY SHRINK?
// -----------
// After deletions, the top levels might become empty.
// Shrinking improves performance by not searching empty levels.
//
// EXAMPLE:
// --------
// Before (after deleting the only height-3 node):
// Level 2: HEAD -> nil (empty!)
// Level 1: HEAD -> [10] -> [20]
// Level 0: HEAD -> [5] -> [10] -> [15] -> [20]
// Height: 3
//
// After shrinking:
// Level 1: HEAD -> [10] -> [20]
// Level 0: HEAD -> [5] -> [10] -> [15] -> [20]
// Height: 2 (top level removed)
func (sl *SkipList) shrink() {
	// Check levels from top down
	for level := sl.Height - 1; level >= 0; level-- {
		if sl.Head.Tower[level] == nil {
			sl.Height-- // This level is empty, reduce height
		} else {
			break // Found a non-empty level, stop
		}
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// RANDOM HEIGHT GENERATION
// ═══════════════════════════════════════════════════════════════════════════════
// This is the "magic" that makes skip lists work!
//
// THE COIN FLIP ALGORITHM:
// -------------------------
// Flip a fair coin repeatedly:
// - Heads: Increase height by 1, flip again
// - Tails: Stop, return current height
//
// PROBABILITY DISTRIBUTION:
// --------------------------
// Height 1: 50% (tails on first flip)
// Height 2: 25% (heads then tails)
// Height 3: 12.5% (heads, heads, tails)
// Height 4: 6.25% (heads, heads, heads, tails)
// ...
//
// This creates a geometric distribution that ensures:
// - Most nodes have height 1 (50%)
// - Few nodes have height 2 (25%)
// - Very few nodes have height 3 (12.5%)
// - Extremely rare to have height > 10
//
// WHY THIS WORKS:
// ---------------
// With N elements and this distribution:
// - Expected number of nodes at level 0: N
// - Expected number of nodes at level 1: N/2
// - Expected number of nodes at level 2: N/4
// - Expected number of nodes at level 3: N/8
// ...
//
// This creates O(log N) expected search time!
// ═══════════════════════════════════════════════════════════════════════════════

// randomHeight generates a random height for a new node
//
// IMPLEMENTATION:
// ---------------
// 1. Start with height = 1
// 2. Flip a coin (random < 0.5)
// 3. If heads and not at max: increase height, repeat
// 4. If tails or at max: return current height
func (sl *SkipList) randomHeight() int {
	height := 1
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Keep "flipping coins" (50% probability)
	for rng.Float64() < 0.5 && height < MaxHeight {
		height++
	}

	return height
}

// ═══════════════════════════════════════════════════════════════════════════════
// ITERATOR: Sequential Access to Elements
// ═══════════════════════════════════════════════════════════════════════════════
// While skip lists support fast random access, sometimes we need to
// traverse all elements in order. The iterator provides this capability.
//
// USAGE PATTERN:
// --------------
// iter := skipList.Iterator()
// for iter.HasNext() {
//     pos := iter.Next()
//     // Process position...
// }
//
// EXAMPLE:
// --------
// Skip list: [Doc1:Pos1, Doc1:Pos5, Doc2:Pos0, Doc2:Pos3]
//
// iter := skipList.Iterator()
// iter.Next() → Doc1:Pos1
// iter.Next() → Doc1:Pos5
// iter.Next() → Doc2:Pos0
// iter.Next() → Doc2:Pos3
// iter.Next() → EOF
// ═══════════════════════════════════════════════════════════════════════════════

// Iterator provides sequential access to skip list elements
//
// IMPLEMENTATION NOTE:
// --------------------
// We only traverse level 0 (the bottom level) which contains all elements
// in sorted order. Higher levels are just shortcuts for searching.
type Iterator struct {
	current *Node // The current position in the iteration
}

// Iterator creates a new iterator starting at the first element
//
// INITIALIZATION:
// ---------------
// We start at the first real element (sl.Head.Tower[0])
// NOT at the Head itself (which is just a sentinel)
func (sl *SkipList) Iterator() *Iterator {
	return &Iterator{current: sl.Head.Tower[0]}
}

// HasNext checks if there are more elements to iterate
//
// LOGIC:
// ------
// There are more elements if:
// - current is not nil (we haven't fallen off the end), AND
// - current.Tower[0] is not nil (there's a next element)
//
// Example states:
// - HasNext() == true:  current -> [next] -> ...
// - HasNext() == false: current -> nil (at the last element)
func (it *Iterator) HasNext() bool {
	return it.current != nil && it.current.Tower[0] != nil
}

// Next advances to and returns the next position
//
// PROCESS:
// --------
// 1. Move to the next node
// 2. If we've reached the end, return EOF
// 3. Otherwise, return the current position
//
// IMPORTANT:
// ----------
// Always check HasNext() before calling Next() to avoid
// returning EOF unexpectedly!
//
// EXAMPLE USAGE:
// --------------
// iter := skipList.Iterator()
//
//	for iter.HasNext() {
//	    pos := iter.Next()
//	    fmt.Printf("Doc %d, Pos %d\n", pos.GetDocumentID(), pos.GetOffset())
//	}
func (it *Iterator) Next() Position {
	// Check if we're already at the end
	if it.current == nil {
		return EOFDocument
	}

	// Move to the next node
	it.current = it.current.Tower[0]

	// Check if we've reached the end after moving
	if it.current == nil {
		return EOFDocument
	}

	// Return the current position
	return it.current.Key
}

// ═══════════════════════════════════════════════════════════════════════════════
// SKIP LIST SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════
//
// KEY CONCEPTS:
// -------------
// 1. Multiple levels: Express lanes for faster searching
// 2. Probabilistic balancing: Random heights keep it balanced on average
// 3. Sorted order: Always maintains elements in sorted order
// 4. O(log n) operations: Search, insert, delete all average O(log n)
//
// WHY IT'S PERFECT FOR SEARCH ENGINES:
// -------------------------------------
// 1. Fast positional lookups: Find any document/position quickly
// 2. Range queries: Find all positions in a document efficiently
// 3. Sorted iteration: Process results in order
// 4. Simple implementation: No complex tree rotations needed
// 5. Good cache performance: Sequential access on level 0
//
// OPERATIONS SUMMARY:
// -------------------
// - Search(key): Find exact key or where it would be → O(log n)
// - Insert(key): Add new element → O(log n)
// - Delete(key): Remove element → O(log n)
// - Find(key): Check if key exists → O(log n)
// - FindLessThan(key): Find predecessor → O(log n)
// - FindGreaterThan(key): Find successor → O(log n)
// - Last(): Find last element → O(n) worst case, O(1) with tail pointer
// - Iterator(): Sequential traversal → O(n) for all elements
//
// SPACE COMPLEXITY:
// -----------------
// - Average: O(n) where n is the number of elements
// - Each node has ~2 pointers on average (geometric distribution)
// - Worst case: O(n * MaxHeight) but extremely unlikely
//
// PERFORMANCE CHARACTERISTICS:
// -----------------------------
// - Search: O(log n) expected, O(n) worst case (very rare)
// - Insert: O(log n) expected, O(n) worst case (very rare)
// - Delete: O(log n) expected, O(n) worst case (very rare)
// - Space: O(n) expected, O(n * log n) worst case
//
// The "worst case" scenarios are so rare they're not practically relevant.
// The randomization ensures good performance with extremely high probability.
//
// COMPARISON TO OTHER DATA STRUCTURES:
// -------------------------------------
// vs. Balanced Trees (AVL, Red-Black):
//   + Simpler implementation (no rotations)
//   + Better constant factors in practice
//   + Lock-free variants easier to implement
//   - Slightly worse worst-case guarantees (probabilistic vs deterministic)
//
// vs. Hash Tables:
//   + Maintains sorted order (hash tables don't)
//   + Supports range queries efficiently
//   + No rehashing needed
//   - Slower than hash tables for exact lookups (O(log n) vs O(1))
//
// vs. Arrays:
//   + Fast insertion/deletion (no shifting elements)
//   + Dynamic sizing (no reallocation)
//   - Slower random access (O(log n) vs O(1))
//   - More memory overhead (pointers)
//
// REAL-WORLD APPLICATIONS:
// -------------------------
// 1. Database indexes (LevelDB, RocksDB use skip lists)
// 2. In-memory caches (Redis sorted sets use skip lists)
// 3. Search engines (inverted indexes like this one!)
// 4. Concurrent data structures (easier to make lock-free than trees)
// 5. Time-series databases (sorted by timestamp)
//
// ═══════════════════════════════════════════════════════════════════════════════
