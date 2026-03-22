package iskra

import (
	"bytes"
	"encoding/binary"
	"math"

	"github.com/RoaringBitmap/roaring"
)

// ═══════════════════════════════════════════════════════════════════════════════
// SERIALIZATION: Saving and Loading the Index
// ═══════════════════════════════════════════════════════════════════════════════
// Why serialize?
// - Save index to disk for persistence
// - Send index over network
// - Create backups
//
// BINARY FORMAT:
// --------------
// We use a custom binary format for efficiency:
// - Smaller file size than JSON (important for large indexes)
// - Faster to parse than JSON
// - Preserves exact structure (including skip list towers)
//
// FORMAT STRUCTURE:
// -----------------
// For each term:
//   [term_length: uint32][term: bytes]
//   [node_data_length: uint32][node_data: positions...]
//   [tower_data: for each node...]
//
// ENCODING STRATEGY:
// ------------------
// The tricky part is encoding the skip list tower structure:
// 1. Assign each node a sequential index (1, 2, 3, ...)
// 2. Store node positions (DocID, Offset pairs)
// 3. Store tower pointers as indices (not memory addresses!)
//
// Why use indices instead of pointers?
// - Pointers are meaningless after deserialization (different memory locations)
// - Indices are stable and can be reconstructed
//
// ═══════════════════════════════════════════════════════════════════════════════

// Encode serializes the inverted index to binary format
//
// COMPLETE EXAMPLE:
// -----------------
// Index contains:
//
//	"quick" → SkipList with nodes at [Doc1:Pos1, Doc3:Pos0]
//	"brown" → SkipList with nodes at [Doc1:Pos2]
//
// Encoded format:
//
//	[5]['q','u','i','c','k']  ← Term name
//	[16][1,1,3,0]              ← Node positions (2 positions × 8 bytes each)
//	[4][2][2][0]               ← Tower structure (node1→node2, node2→nil)
//	[5]['b','r','o','w','n']  ← Next term
//	[8][1,2]                   ← Node position
//	[2][0]                     ← Tower structure (only one node, no next)
//
// The encoder object keeps track of our position in the output buffer.
// Encode serializes the inverted index with HYBRID STORAGE including BM25 statistics
//
// BINARY FORMAT:
// --------------
// [Header]
//   - TotalDocs: uint32
//   - TotalTerms: uint64
//   - BM25.K1: float64
//   - BM25.B: float64
//   - NumDocStats: uint32
//
// [Document Statistics] (for each document)
//   - DocID: uint32
//   - Length: uint32
//   - NumTerms: uint32
//   - For each term:
//   - TermLength: uint32
//   - Term: bytes
//   - Frequency: uint32
//
// [Roaring Bitmaps] (NEW - for fast document lookups)
//   - NumBitmaps: uint32
//   - For each term:
//   - TermLength: uint32
//   - Term: bytes
//   - BitmapLength: uint32
//   - Bitmap: bytes (roaring's native serialization)
//
// [Posting Lists] (position data for phrase search)
//   - For each term...
func (idx *InvertedIndex) Encode() ([]byte, error) {
	buf := new(bytes.Buffer)

	// Write header with BM25 metadata
	if err := idx.encodeHeader(buf); err != nil {
		return nil, err
	}

	// Write document statistics
	if err := idx.encodeDocStats(buf); err != nil {
		return nil, err
	}

	// Write roaring bitmaps (NEW!)
	if err := idx.encodeRoaringBitmaps(buf); err != nil {
		return nil, err
	}

	// Write posting lists (existing format)
	encoder := newIndexEncoder(buf)
	for term, skipList := range idx.PostingsList {
		if err := encoder.encodeTerm(term, skipList); err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// encodeHeader writes the index metadata
func (idx *InvertedIndex) encodeHeader(buf *bytes.Buffer) error {
	// Write corpus statistics
	if err := binary.Write(buf, binary.LittleEndian, uint32(idx.TotalDocs)); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint64(idx.TotalTerms)); err != nil {
		return err
	}

	// Write BM25 parameters
	if err := binary.Write(buf, binary.LittleEndian, idx.BM25Params.K1); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, idx.BM25Params.B); err != nil {
		return err
	}

	// Write number of documents with statistics
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(idx.DocStats))); err != nil {
		return err
	}

	return nil
}

// encodeDocStats writes document statistics for BM25
func (idx *InvertedIndex) encodeDocStats(buf *bytes.Buffer) error {
	for _, docStats := range idx.DocStats {
		// Write document ID and length
		if err := binary.Write(buf, binary.LittleEndian, uint32(docStats.DocID)); err != nil {
			return err
		}
		if err := binary.Write(buf, binary.LittleEndian, uint32(docStats.Length)); err != nil {
			return err
		}

		// Write number of unique terms
		if err := binary.Write(buf, binary.LittleEndian, uint32(len(docStats.TermFreqs))); err != nil {
			return err
		}

		// Write each term and its frequency
		for term, freq := range docStats.TermFreqs {
			// Write term length and term
			termBytes := []byte(term)
			if err := binary.Write(buf, binary.LittleEndian, uint32(len(termBytes))); err != nil {
				return err
			}
			if _, err := buf.Write(termBytes); err != nil {
				return err
			}

			// Write frequency
			if err := binary.Write(buf, binary.LittleEndian, uint32(freq)); err != nil {
				return err
			}
		}
	}

	return nil
}

// encodeRoaringBitmaps writes the roaring bitmaps for document-level storage
//
// ROARING BITMAP SERIALIZATION:
// ------------------------------
// Roaring bitmaps have their own efficient binary format via ToBytes()
// We just need to wrap it with term names and lengths
//
// FORMAT:
// -------
// [NumBitmaps: uint32]
// For each bitmap:
//
//	[TermLength: uint32][Term: bytes]
//	[BitmapLength: uint32][Bitmap: bytes]
//
// EXAMPLE:
// --------
// Term "quick" appears in documents [1, 3, 5, 100, 500]
// Roaring serializes this to ~20 bytes (vs 40 bytes for raw integers!)
//
// COMPRESSION BENEFITS:
// ---------------------
// For term "the" in 500,000 documents:
// - Skip list: ~24 MB (500k nodes × 48 bytes)
// - Roaring bitmap: ~60 KB (400x compression!)
func (idx *InvertedIndex) encodeRoaringBitmaps(buf *bytes.Buffer) error {
	// Write number of bitmaps
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(idx.DocBitmaps))); err != nil {
		return err
	}

	// Write each term and its bitmap
	for term, bitmap := range idx.DocBitmaps {
		// Write term name
		termBytes := []byte(term)
		if err := binary.Write(buf, binary.LittleEndian, uint32(len(termBytes))); err != nil {
			return err
		}
		if _, err := buf.Write(termBytes); err != nil {
			return err
		}

		// Write roaring bitmap (it has its own compact serialization)
		bitmapBytes, err := bitmap.ToBytes()
		if err != nil {
			return err
		}
		if err := binary.Write(buf, binary.LittleEndian, uint32(len(bitmapBytes))); err != nil {
			return err
		}
		if _, err := buf.Write(bitmapBytes); err != nil {
			return err
		}
	}

	return nil
}

// indexEncoder handles the encoding process
//
// This encapsulates the encoding state and provides helper methods.
// Using a struct is cleaner than passing a buffer around everywhere.
type indexEncoder struct {
	buffer *bytes.Buffer // Accumulates the serialized data
}

func newIndexEncoder(buffer *bytes.Buffer) *indexEncoder {
	return &indexEncoder{
		buffer: buffer,
	}
}

// encodeTerm serializes a single term and its skip list
//
// THREE-PHASE ENCODING:
// ---------------------
// Phase 1: Write the term name
// Phase 2: Write node positions (DocID, Offset pairs)
// Phase 3: Write tower structure (how nodes link together)
func (e *indexEncoder) encodeTerm(term string, skipList SkipList) error {
	// PHASE 1: Write term name
	// Format: [length: uint32][bytes]
	if err := e.writeString(term); err != nil {
		return err
	}

	// PHASE 2: Build node index map
	// Assign each node a sequential index: Head=1, Next=2, etc.
	// This map lets us convert node pointers to indices
	nodeMap := e.buildNodeIndexMap(skipList)

	// PHASE 3: Write node positions
	// Format: [length: uint32][DocID: uint32][Offset: uint32]...
	nodeData := e.encodeNodePositions(skipList)
	if err := e.writeBytes(nodeData); err != nil {
		return err
	}

	// PHASE 4: Write tower structure
	// This is the most complex part - see encodeTowerStructure
	return e.encodeTowerStructure(skipList, nodeMap)
}

// writeString writes a length-prefixed string
//
// Format: [length: 4 bytes][string: length bytes]
//
// Example: "quick" (5 characters)
//
//	Binary: [0x05, 0x00, 0x00, 0x00, 'q', 'u', 'i', 'c', 'k']
//	         ^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
//	         length = 5            actual string bytes
func (e *indexEncoder) writeString(s string) error {
	data := []byte(s)

	// Write length as 32-bit unsigned integer (4 bytes)
	if err := binary.Write(e.buffer, binary.LittleEndian, uint32(len(data))); err != nil {
		return err
	}

	// Write the actual string bytes
	_, err := e.buffer.Write(data)
	return err
}

// writeBytes writes a length-prefixed byte array
//
// Same as writeString but for arbitrary byte data
func (e *indexEncoder) writeBytes(data []byte) error {
	// Write length prefix
	if err := binary.Write(e.buffer, binary.LittleEndian, uint32(len(data))); err != nil {
		return err
	}

	// Write the data
	_, err := e.buffer.Write(data)
	return err
}

// buildNodeIndexMap creates a mapping from node positions to sequential indices
//
// WHY DO WE NEED THIS?
// --------------------
// Skip list nodes are connected via pointers (memory addresses).
// We can't serialize pointers because:
// 1. Memory addresses change between program runs
// 2. Addresses are meaningless on different machines
//
// Solution: Assign each node a stable index (1, 2, 3, ...)
// Then we can say "Node 1 points to Node 3" instead of memory addresses.
//
// EXAMPLE:
// --------
// Skip list: Head → Node{Doc1:Pos1} → Node{Doc3:Pos0} → nil
//
// Build map:
//
//	{Doc1:Pos1} → Index 1
//	{Doc3:Pos0} → Index 2
//
// Now we can encode towers as: "Node 1 points to Index 2"
func (e *indexEncoder) buildNodeIndexMap(skipList SkipList) map[nodePosition]int {
	nodeMap := make(map[nodePosition]int)
	current := skipList.Head
	index := 1 // Start from 1 (0 means nil/null)

	// Traverse the bottom level of the skip list
	for current != nil {
		// Create a compact position identifier
		pos := nodePosition{
			DocID:    int32(current.Key.DocumentID),
			Position: int32(current.Key.Offset),
		}

		// Assign this node the next sequential index
		nodeMap[pos] = index
		index++

		// Move to next node
		current = current.Tower[0]
	}

	return nodeMap
}

// encodeNodePositions serializes all node positions (DocID, Offset pairs)
//
// FORMAT:
// -------
// For each node: [DocID: int32][Offset: int32]
//
// EXAMPLE:
// --------
// Nodes: [Doc1:Pos1, Doc3:Pos0, Doc3:Pos5]
// Encoded: [1][1][3][0][3][5]
//
//	^^^  ^^^  ^^^  ^^^  ^^^  ^^^
//	DocID Off DocID Off DocID Off
//
// Total: 6 × 4 bytes = 24 bytes
func (e *indexEncoder) encodeNodePositions(skipList SkipList) []byte {
	buf := new(bytes.Buffer)
	current := skipList.Head

	// Traverse all nodes in the skip list
	for current != nil {
		// Write document ID (4 bytes)
		binary.Write(buf, binary.LittleEndian, int32(current.Key.DocumentID))

		// Write offset (4 bytes)
		binary.Write(buf, binary.LittleEndian, int32(current.Key.Offset))

		// Move to next node
		current = current.Tower[0]
	}

	return buf.Bytes()
}

// encodeTowerStructure serializes the skip list tower connections
//
// TOWER STRUCTURE RECAP:
// ----------------------
// A skip list node has a "tower" of forward pointers at different levels:
//
//	Level 2: [*]---------------→[*]----------→nil
//	Level 1: [*]------→[*]------→[*]------→[*]→nil
//	Level 0: [*]→[*]→[*]→[*]→[*]→[*]→[*]→[*]→nil
//
// Each node's tower is an array of pointers to other nodes.
//
// ENCODING STRATEGY:
// ------------------
// For each node, we encode which nodes its tower points to (as indices).
//
// EXAMPLE:
// --------
// Node 1 tower: [Node2, Node4, nil, nil, ...] (2 levels high)
// Node 2 tower: [Node3, nil, nil, ...]        (1 level high)
// Node 3 tower: [Node4, nil, nil, ...]        (1 level high)
//
// Encoded:
//
//	Node 1: [length=4][2][4]      ← 2 indices × 2 bytes = 4 bytes
//	Node 2: [length=2][3]          ← 1 index × 2 bytes = 2 bytes
//	Node 3: [length=2][4]          ← 1 index × 2 bytes = 2 bytes
func (e *indexEncoder) encodeTowerStructure(skipList SkipList, nodeMap map[nodePosition]int) error {
	current := skipList.Head

	// Encode tower for each node in the skip list
	for current != nil {
		towerData := e.encodeTowerForNode(current, nodeMap)
		if err := e.writeBytes(towerData); err != nil {
			return err
		}
		current = current.Tower[0]
	}

	return nil
}

// encodeTowerForNode encodes the tower structure for a single node
//
// PROCESS:
// --------
// 1. Collect all non-nil tower pointers
// 2. Convert each pointer to its index (using nodeMap)
// 3. Write indices as uint16 values
//
// Special case: If tower is empty (no forward pointers), write [0]
func (e *indexEncoder) encodeTowerForNode(node *Node, nodeMap map[nodePosition]int) []byte {
	buf := new(bytes.Buffer)

	// Collect all non-nil tower levels
	towerIndices := e.collectTowerIndices(node, nodeMap)

	if len(towerIndices) == 0 {
		// Empty tower - write a single zero
		binary.Write(buf, binary.LittleEndian, uint16(0))
	} else {
		// Write each index as a 2-byte value
		for _, index := range towerIndices {
			binary.Write(buf, binary.LittleEndian, uint16(index))
		}
	}

	return buf.Bytes()
}

// collectTowerIndices extracts tower pointers and converts them to indices
//
// WALK THROUGH:
// -------------
// Given a node with tower: [PtrA, PtrB, nil, nil, ...]
//
// Step 1: Check level 0 - PtrA exists
//   - Look up PtrA's position in nodeMap
//   - Get index: 3
//   - Add 3 to indices
//
// Step 2: Check level 1 - PtrB exists
//   - Look up PtrB's position in nodeMap
//   - Get index: 7
//   - Add 7 to indices
//
// Step 3: Check level 2 - nil
//   - Stop here
//
// Result: [3, 7]
func (e *indexEncoder) collectTowerIndices(node *Node, nodeMap map[nodePosition]int) []int {
	var indices []int

	// Walk up the tower until we hit a nil pointer
	for level := 0; level < MaxHeight; level++ {
		if node.Tower[level] == nil {
			break // No more levels
		}

		// Get the position of the target node
		pos := nodePosition{
			DocID:    int32(node.Tower[level].Key.DocumentID),
			Position: int32(node.Tower[level].Key.Offset),
		}

		// Look up the target node's index
		indices = append(indices, nodeMap[pos])
	}

	return indices
}

// nodePosition represents a compact node position for encoding
//
// We use int32 to match our internal representation:
// - Document IDs are integers
// - Positions are integers
// - Sentinel values (BOF/EOF) use int as well
//
// int32 provides:
// - 4 bytes per value (efficient storage)
// - Range: -2,147,483,648 to 2,147,483,647
// - Sufficient for document IDs and positions
type nodePosition struct {
	DocID    int32
	Position int32
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESERIALIZATION: Loading the Index from Binary Data
// ═══════════════════════════════════════════════════════════════════════════════
// This is the reverse of encoding - we read the binary data and reconstruct
// the entire index structure including all skip list pointers.
//
// THREE-PHASE DECODING:
// ---------------------
// Phase 1: Read term names and node positions
// Phase 2: Create node objects
// Phase 3: Reconstruct tower pointers (the tricky part!)
//
// ═══════════════════════════════════════════════════════════════════════════════

// Decode deserializes binary data back into an inverted index
//
// PROCESS:
// --------
// 1. Create a decoder to track our position in the byte array
// 2. Repeatedly decode terms until we reach the end
// 3. Reconstruct the PostingsList map
//
// EXAMPLE:
// --------
// Input: [5]['quick'][16][1,1,3,0][4][2][2][0]...
// Output: PostingsList["quick"] = SkipList{...}
// Decode deserializes binary data back into an inverted index with HYBRID STORAGE and BM25 stats
func (idx *InvertedIndex) Decode(data []byte) error {
	offset := 0

	// Read header with BM25 metadata
	newOffset, err := idx.decodeHeader(data, offset)
	if err != nil {
		return err
	}
	offset = newOffset

	// Read document statistics
	newOffset, err = idx.decodeDocStats(data, offset)
	if err != nil {
		return err
	}
	offset = newOffset

	// Read roaring bitmaps (NEW!)
	newOffset, err = idx.decodeRoaringBitmaps(data, offset)
	if err != nil {
		return err
	}
	offset = newOffset

	// Read posting lists (existing format)
	decoder := newIndexDecoder(data, offset)
	recoveredIndex := make(map[string]SkipList)

	for !decoder.isComplete() {
		term, skipList, err := decoder.decodeTerm()
		if err != nil {
			return err
		}
		recoveredIndex[term] = skipList
	}

	idx.PostingsList = recoveredIndex
	return nil
}

// decodeHeader reads the index metadata
func (idx *InvertedIndex) decodeHeader(data []byte, offset int) (int, error) {
	// Read corpus statistics
	idx.TotalDocs = int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	idx.TotalTerms = int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
	offset += 8

	// Read BM25 parameters
	idx.BM25Params.K1 = math.Float64frombits(binary.LittleEndian.Uint64(data[offset : offset+8]))
	offset += 8

	idx.BM25Params.B = math.Float64frombits(binary.LittleEndian.Uint64(data[offset : offset+8]))
	offset += 8

	return offset, nil
}

// decodeDocStats reads document statistics
func (idx *InvertedIndex) decodeDocStats(data []byte, offset int) (int, error) {
	// Read number of documents
	numDocs := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	idx.DocStats = make(map[int]DocumentStats, numDocs)

	for i := 0; i < numDocs; i++ {
		// Read document ID and length
		docID := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		length := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		// Read number of unique terms
		numTerms := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		// Initialize document stats
		docStats := DocumentStats{
			DocID:     docID,
			Length:    length,
			TermFreqs: make(map[string]int, numTerms),
		}

		// Read each term and its frequency
		for j := 0; j < numTerms; j++ {
			// Read term length
			termLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4

			// Read term
			term := string(data[offset : offset+termLen])
			offset += termLen

			// Read frequency
			freq := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4

			docStats.TermFreqs[term] = freq
		}

		idx.DocStats[docID] = docStats
	}

	return offset, nil
}

// decodeRoaringBitmaps reads the roaring bitmaps for document-level storage
//
// DESERIALIZATION:
// ----------------
// Read each term and its roaring bitmap, reconstructing the DocBitmaps map
//
// FORMAT:
// -------
// [NumBitmaps: uint32]
// For each bitmap:
//
//	[TermLength: uint32][Term: bytes]
//	[BitmapLength: uint32][Bitmap: bytes]
//
// RECOVERY:
// ---------
// We create a new roaring.Bitmap for each term and deserialize it
// using roaring's native UnmarshalBinary method
func (idx *InvertedIndex) decodeRoaringBitmaps(data []byte, offset int) (int, error) {
	// Read number of bitmaps
	numBitmaps := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	// Initialize the DocBitmaps map
	idx.DocBitmaps = make(map[string]*roaring.Bitmap, numBitmaps)

	// Read each term and its bitmap
	for i := 0; i < numBitmaps; i++ {
		// Read term length
		termLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		// Read term
		term := string(data[offset : offset+termLen])
		offset += termLen

		// Read bitmap length
		bitmapLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		// Read and deserialize bitmap
		bitmap := roaring.NewBitmap()
		if err := bitmap.UnmarshalBinary(data[offset : offset+bitmapLen]); err != nil {
			return 0, err
		}
		offset += bitmapLen

		// Store in map
		idx.DocBitmaps[term] = bitmap
	}

	return offset, nil
}

// indexDecoder handles the decoding process
//
// State management:
// - data: The full byte array we're decoding
// - offset: Our current position in the array
type indexDecoder struct {
	data   []byte
	offset int
}

func newIndexDecoder(data []byte, offset int) *indexDecoder {
	return &indexDecoder{
		data:   data,
		offset: offset,
	}
}

// isComplete checks if we've decoded all the data
func (d *indexDecoder) isComplete() bool {
	return d.offset >= len(d.data)
}

// decodeTerm decodes a single term and its skip list
//
// DECODING SEQUENCE:
// ------------------
// 1. Read term name: "quick"
// 2. Read node positions: [Doc1:Pos1, Doc3:Pos0]
// 3. Create node objects with these positions
// 4. Read tower structure and link nodes together
// 5. Return the reconstructed SkipList
func (d *indexDecoder) decodeTerm() (string, SkipList, error) {
	// Step 1: Read the term name
	term, err := d.readString()
	if err != nil {
		return "", SkipList{}, err
	}

	// Step 2: Read and decode node positions
	// Returns a map: Index → Node pointer
	nodeMap, err := d.decodeNodePositions()
	if err != nil {
		return "", SkipList{}, err
	}

	// Step 3: Decode tower structure (reconnect the nodes)
	height, err := d.decodeTowerStructure(nodeMap)
	if err != nil {
		return "", SkipList{}, err
	}

	// Step 4: Create the SkipList structure
	skipList := SkipList{
		Head:   nodeMap[1], // First node is always at index 1
		Height: height,
	}

	return term, skipList, nil
}

// readString reads a length-prefixed string
//
// Format: [length: 4 bytes][string: length bytes]
//
// EXAMPLE:
// --------
// Data: [0x05, 0x00, 0x00, 0x00, 'q', 'u', 'i', 'c', 'k', ...]
//
//	^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^
//	length = 5            string bytes
//
// Returns: "quick"
// Advances offset by: 4 + 5 = 9 bytes
func (d *indexDecoder) readString() (string, error) {
	// Read the length (4 bytes)
	length := int(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
	d.offset += 4

	// Read the string bytes
	str := string(d.data[d.offset : d.offset+length])
	d.offset += length

	return str, nil
}

// decodeNodePositions reconstructs all nodes from their serialized positions
//
// INPUT FORMAT:
// -------------
// [dataLength: 4 bytes][DocID: 4 bytes][Offset: 4 bytes]...
//
// PROCESS:
// --------
// 1. Read data length: How many bytes of position data?
// 2. Calculate number of values: dataLength / 4
// 3. Read pairs of values: (DocID, Offset)
// 4. Create Node objects
// 5. Assign sequential indices: 1, 2, 3, ...
//
// EXAMPLE:
// --------
// Data: [16][1][1][3][0]
//
//	^^^ 16 bytes of position data
//	    ^^ DocID=1, Offset=1 → Node 1
//	          ^^ DocID=3, Offset=0 → Node 2
//
// Result: map[1→Node{Doc1:Pos1}, 2→Node{Doc3:Pos0}]
func (d *indexDecoder) decodeNodePositions() (map[int]*Node, error) {
	// Read the length of position data
	dataLength := int(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
	d.offset += 4

	nodeMap := make(map[int]*Node)
	nodeIndex := 1

	// Each position is 8 bytes: 4 for DocID + 4 for Offset
	// So numValues = dataLength / 4 gives us the total number of int32s
	// And we process them in pairs
	numValues := dataLength / 4

	for i := 0; i < numValues; i += 2 {
		// Read Document ID (as int32)
		docID := int32(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
		d.offset += 4

		// Read Offset (as int32)
		offset := int32(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
		d.offset += 4

		// Create a new node with this position
		node := &Node{
			Key: Position{
				DocumentID: int(docID),
				Offset:     int(offset),
			},
		}

		// Assign it a sequential index
		nodeMap[nodeIndex] = node
		nodeIndex++
	}

	return nodeMap, nil
}

// decodeTowerStructure reconstructs the skip list tower connections
//
// THIS IS THE MAGIC STEP!
// -----------------------
// We now have nodes, but they're not connected.
// This function reads the tower indices and reconnects everything.
//
// INPUT FORMAT (for each node):
// -----------------------------
// [towerLength: 4 bytes][index1: 2 bytes][index2: 2 bytes]...
//
// EXAMPLE:
// --------
// Node 1: [4][2][4]  ← Tower has 2 levels: points to nodes 2 and 4
// Node 2: [2][3]      ← Tower has 1 level: points to node 3
// Node 3: [2][0]      ← Tower has 1 level: points to nothing (end)
//
// RECONSTRUCTION:
// ---------------
// For Node 1:
//   - Read indices: [2, 4]
//   - Set Tower[0] = nodeMap[2]
//   - Set Tower[1] = nodeMap[4]
//
// Result: Node 1 is now connected to nodes 2 and 4 at levels 0 and 1!
func (d *indexDecoder) decodeTowerStructure(nodeMap map[int]*Node) (int, error) {
	maxHeight := 1 // Track the maximum tower height
	nodeCount := len(nodeMap)

	// Process tower data for each node
	for nodeIndex := 1; nodeIndex <= nodeCount; nodeIndex++ {
		// Read the length of tower data for this node
		towerLength := int(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
		d.offset += 4

		// Calculate how many indices are stored (each index is 2 bytes)
		numIndices := towerLength / 2

		// Read each tower level
		for level := 0; level < numIndices; level++ {
			// Read the target node index
			targetIndex := int(binary.LittleEndian.Uint16(d.data[d.offset : d.offset+2]))
			d.offset += 2

			// If index is not 0 (0 means nil), connect the nodes
			if targetIndex != 0 {
				nodeMap[nodeIndex].Tower[level] = nodeMap[targetIndex]

				// Track maximum height
				if level+1 > maxHeight {
					maxHeight = level + 1
				}
			}
		}
	}

	return maxHeight, nil
}
