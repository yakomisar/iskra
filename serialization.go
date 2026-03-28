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
//
// BINARY FORMAT:
// --------------
// [Header]
//   TotalDocs  uint32
//   TotalTerms uint64
//   BM25.K1    float64
//   BM25.B     float64
//   NumDocStats uint32
//
// [Document Statistics] (repeated NumDocStats times)
//   DocID      uint32
//   Length     uint32
//   NumTerms   uint32
//   For each term:
//     TermLength uint32
//     Term       bytes
//     Frequency  uint32
//
// [Roaring Bitmaps]
//   NumBitmaps uint32
//   For each:
//     TermLength   uint32
//     Term         bytes
//     BitmapLength uint32
//     Bitmap       bytes
//
// [Posting Lists]
//   For each term:
//     TermLength  uint32
//     Term        bytes
//     NodeData    length-prefixed (DocID uint32, Offset int32 pairs)
//     TowerData   length-prefixed per node
//
// NOTE: DocID in node positions is encoded as uint32 (was int32 before).
// ═══════════════════════════════════════════════════════════════════════════════

// Encode serializes the inverted index to binary format
func (idx *InvertedIndex) Encode() ([]byte, error) {
	buf := new(bytes.Buffer)

	if err := idx.encodeHeader(buf); err != nil {
		return nil, err
	}

	if err := idx.encodeDocStats(buf); err != nil {
		return nil, err
	}

	if err := idx.encodeRoaringBitmaps(buf); err != nil {
		return nil, err
	}

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
	if err := binary.Write(buf, binary.LittleEndian, uint32(idx.TotalDocs)); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint64(idx.TotalTerms)); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, idx.BM25Params.K1); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, idx.BM25Params.B); err != nil {
		return err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(idx.DocStats))); err != nil {
		return err
	}
	return nil
}

// encodeDocStats writes document statistics for BM25
func (idx *InvertedIndex) encodeDocStats(buf *bytes.Buffer) error {
	for _, docStats := range idx.DocStats {
		// DocID is already uint32
		if err := binary.Write(buf, binary.LittleEndian, docStats.DocID); err != nil {
			return err
		}
		if err := binary.Write(buf, binary.LittleEndian, uint32(docStats.Length)); err != nil {
			return err
		}
		if err := binary.Write(buf, binary.LittleEndian, uint32(len(docStats.TermFreqs))); err != nil {
			return err
		}
		for term, freq := range docStats.TermFreqs {
			termBytes := []byte(term)
			if err := binary.Write(buf, binary.LittleEndian, uint32(len(termBytes))); err != nil {
				return err
			}
			if _, err := buf.Write(termBytes); err != nil {
				return err
			}
			if err := binary.Write(buf, binary.LittleEndian, uint32(freq)); err != nil {
				return err
			}
		}
	}
	return nil
}

// encodeRoaringBitmaps writes the roaring bitmaps for document-level storage
func (idx *InvertedIndex) encodeRoaringBitmaps(buf *bytes.Buffer) error {
	if err := binary.Write(buf, binary.LittleEndian, uint32(len(idx.DocBitmaps))); err != nil {
		return err
	}

	for term, bitmap := range idx.DocBitmaps {
		termBytes := []byte(term)
		if err := binary.Write(buf, binary.LittleEndian, uint32(len(termBytes))); err != nil {
			return err
		}
		if _, err := buf.Write(termBytes); err != nil {
			return err
		}

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

// ═══════════════════════════════════════════════════════════════════════════════
// indexEncoder
// ═══════════════════════════════════════════════════════════════════════════════

type indexEncoder struct {
	buffer *bytes.Buffer
}

func newIndexEncoder(buffer *bytes.Buffer) *indexEncoder {
	return &indexEncoder{buffer: buffer}
}

func (e *indexEncoder) encodeTerm(term string, skipList SkipList) error {
	if err := e.writeString(term); err != nil {
		return err
	}

	nodeMap := e.buildNodeIndexMap(skipList)

	nodeData := e.encodeNodePositions(skipList)
	if err := e.writeBytes(nodeData); err != nil {
		return err
	}

	return e.encodeTowerStructure(skipList, nodeMap)
}

func (e *indexEncoder) writeString(s string) error {
	data := []byte(s)
	if err := binary.Write(e.buffer, binary.LittleEndian, uint32(len(data))); err != nil {
		return err
	}
	_, err := e.buffer.Write(data)
	return err
}

func (e *indexEncoder) writeBytes(data []byte) error {
	if err := binary.Write(e.buffer, binary.LittleEndian, uint32(len(data))); err != nil {
		return err
	}
	_, err := e.buffer.Write(data)
	return err
}

// buildNodeIndexMap assigns sequential indices to skip list nodes.
// nodePosition now uses uint32 for DocID to match Position.DocumentID.
func (e *indexEncoder) buildNodeIndexMap(skipList SkipList) map[nodePosition]int {
	nodeMap := make(map[nodePosition]int)
	current := skipList.Head
	index := 1

	for current != nil {
		pos := nodePosition{
			DocID:    current.Key.DocumentID, // uint32
			Position: int32(current.Key.Offset),
		}
		nodeMap[pos] = index
		index++
		current = current.Tower[0]
	}
	return nodeMap
}

// encodeNodePositions serializes node positions as [DocID uint32][Offset int32] pairs.
// DocID is now written as uint32 (previously int32).
func (e *indexEncoder) encodeNodePositions(skipList SkipList) []byte {
	buf := new(bytes.Buffer)
	current := skipList.Head

	for current != nil {
		// DocID as uint32
		binary.Write(buf, binary.LittleEndian, current.Key.DocumentID)
		// Offset as int32 (still signed, carries BOF/EOF sentinels)
		binary.Write(buf, binary.LittleEndian, int32(current.Key.Offset))
		current = current.Tower[0]
	}
	return buf.Bytes()
}

func (e *indexEncoder) encodeTowerStructure(skipList SkipList, nodeMap map[nodePosition]int) error {
	current := skipList.Head
	for current != nil {
		towerData := e.encodeTowerForNode(current, nodeMap)
		if err := e.writeBytes(towerData); err != nil {
			return err
		}
		current = current.Tower[0]
	}
	return nil
}

func (e *indexEncoder) encodeTowerForNode(node *Node, nodeMap map[nodePosition]int) []byte {
	buf := new(bytes.Buffer)
	towerIndices := e.collectTowerIndices(node, nodeMap)

	if len(towerIndices) == 0 {
		binary.Write(buf, binary.LittleEndian, uint16(0))
	} else {
		for _, index := range towerIndices {
			binary.Write(buf, binary.LittleEndian, uint16(index))
		}
	}
	return buf.Bytes()
}

func (e *indexEncoder) collectTowerIndices(node *Node, nodeMap map[nodePosition]int) []int {
	var indices []int
	for level := 0; level < MaxHeight; level++ {
		if node.Tower[level] == nil {
			break
		}
		pos := nodePosition{
			DocID:    node.Tower[level].Key.DocumentID,
			Position: int32(node.Tower[level].Key.Offset),
		}
		indices = append(indices, nodeMap[pos])
	}
	return indices
}

// nodePosition is the compact key used in the encoder's node map.
// DocID is uint32 to match Position.DocumentID.
type nodePosition struct {
	DocID    uint32
	Position int32
}

// ═══════════════════════════════════════════════════════════════════════════════
// DESERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

// Decode deserializes binary data back into an inverted index
func (idx *InvertedIndex) Decode(data []byte) error {
	offset := 0

	newOffset, err := idx.decodeHeader(data, offset)
	if err != nil {
		return err
	}
	offset = newOffset

	newOffset, err = idx.decodeDocStats(data, offset)
	if err != nil {
		return err
	}
	offset = newOffset

	newOffset, err = idx.decodeRoaringBitmaps(data, offset)
	if err != nil {
		return err
	}
	offset = newOffset

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
	idx.TotalDocs = int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	idx.TotalTerms = int64(binary.LittleEndian.Uint64(data[offset : offset+8]))
	offset += 8

	idx.BM25Params.K1 = math.Float64frombits(binary.LittleEndian.Uint64(data[offset : offset+8]))
	offset += 8

	idx.BM25Params.B = math.Float64frombits(binary.LittleEndian.Uint64(data[offset : offset+8]))
	offset += 8

	return offset, nil
}

// decodeDocStats reads document statistics.
// DocID is read as uint32.
func (idx *InvertedIndex) decodeDocStats(data []byte, offset int) (int, error) {
	numDocs := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	idx.DocStats = make(map[uint32]DocumentStats, numDocs)

	for i := 0; i < numDocs; i++ {
		docID := binary.LittleEndian.Uint32(data[offset : offset+4])
		offset += 4

		length := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		numTerms := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		docStats := DocumentStats{
			DocID:     docID,
			Length:    length,
			TermFreqs: make(map[string]int, numTerms),
		}

		for j := 0; j < numTerms; j++ {
			termLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4

			term := string(data[offset : offset+termLen])
			offset += termLen

			freq := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
			offset += 4

			docStats.TermFreqs[term] = freq
		}

		idx.DocStats[docID] = docStats
	}

	return offset, nil
}

// decodeRoaringBitmaps reads the roaring bitmaps for document-level storage
func (idx *InvertedIndex) decodeRoaringBitmaps(data []byte, offset int) (int, error) {
	numBitmaps := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
	offset += 4

	idx.DocBitmaps = make(map[string]*roaring.Bitmap, numBitmaps)

	for i := 0; i < numBitmaps; i++ {
		termLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		term := string(data[offset : offset+termLen])
		offset += termLen

		bitmapLen := int(binary.LittleEndian.Uint32(data[offset : offset+4]))
		offset += 4

		bitmap := roaring.NewBitmap()
		if err := bitmap.UnmarshalBinary(data[offset : offset+bitmapLen]); err != nil {
			return 0, err
		}
		offset += bitmapLen

		idx.DocBitmaps[term] = bitmap
	}

	return offset, nil
}

// ═══════════════════════════════════════════════════════════════════════════════
// indexDecoder
// ═══════════════════════════════════════════════════════════════════════════════

type indexDecoder struct {
	data   []byte
	offset int
}

func newIndexDecoder(data []byte, offset int) *indexDecoder {
	return &indexDecoder{data: data, offset: offset}
}

func (d *indexDecoder) isComplete() bool {
	return d.offset >= len(d.data)
}

func (d *indexDecoder) decodeTerm() (string, SkipList, error) {
	term, err := d.readString()
	if err != nil {
		return "", SkipList{}, err
	}

	nodeMap, err := d.decodeNodePositions()
	if err != nil {
		return "", SkipList{}, err
	}

	height, err := d.decodeTowerStructure(nodeMap)
	if err != nil {
		return "", SkipList{}, err
	}

	skipList := SkipList{
		Head:   nodeMap[1],
		Height: height,
	}

	return term, skipList, nil
}

func (d *indexDecoder) readString() (string, error) {
	length := int(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
	d.offset += 4

	str := string(d.data[d.offset : d.offset+length])
	d.offset += length

	return str, nil
}

// decodeNodePositions reconstructs nodes from serialized [DocID uint32][Offset int32] pairs.
// DocID is now read as uint32.
func (d *indexDecoder) decodeNodePositions() (map[int]*Node, error) {
	dataLength := int(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
	d.offset += 4

	nodeMap := make(map[int]*Node)
	nodeIndex := 1

	// Each position pair: 4 bytes (uint32 DocID) + 4 bytes (int32 Offset) = 8 bytes
	numValues := dataLength / 4
	for i := 0; i < numValues; i += 2 {
		docID := binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4])
		d.offset += 4

		offset := int32(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
		d.offset += 4

		node := &Node{
			Key: Position{
				DocumentID: docID,
				Offset:     int(offset),
			},
		}
		nodeMap[nodeIndex] = node
		nodeIndex++
	}

	return nodeMap, nil
}

func (d *indexDecoder) decodeTowerStructure(nodeMap map[int]*Node) (int, error) {
	maxHeight := 1
	nodeCount := len(nodeMap)

	for nodeIndex := 1; nodeIndex <= nodeCount; nodeIndex++ {
		towerLength := int(binary.LittleEndian.Uint32(d.data[d.offset : d.offset+4]))
		d.offset += 4

		numIndices := towerLength / 2

		for level := 0; level < numIndices; level++ {
			targetIndex := int(binary.LittleEndian.Uint16(d.data[d.offset : d.offset+2]))
			d.offset += 2

			if targetIndex != 0 {
				nodeMap[nodeIndex].Tower[level] = nodeMap[targetIndex]
				if level+1 > maxHeight {
					maxHeight = level + 1
				}
			}
		}
	}

	return maxHeight, nil
}
