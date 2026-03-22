// ═══════════════════════════════════════════════════════════════════════════════
// TEXT ANALYSIS OVERVIEW
// ═══════════════════════════════════════════════════════════════════════════════
// Text analysis transforms raw text into searchable tokens through a multi-stage
// pipeline. This process is crucial for effective full-text search.
//
// ANALYSIS PIPELINE:
// ------------------
//  1. Tokenization   → Split text into words
//  2. Lowercasing    → Normalize case ("Quick" → "quick")
//  3. Stop word removal → Remove common words ("the", "a", etc.)
//  4. Length filtering  → Remove very short tokens (< 2 chars)
//  5. Stemming       → Reduce words to root form ("running" → "run")
//
// EXAMPLE TRANSFORMATION:
// -----------------------
// Input:  "The Quick Brown Fox Jumps!"
// Step 1: ["The", "Quick", "Brown", "Fox", "Jumps"]     (tokenize)
// Step 2: ["the", "quick", "brown", "fox", "jumps"]     (lowercase)
// Step 3: ["quick", "brown", "fox", "jumps"]            (remove stopwords)
// Step 4: ["quick", "brown", "fox", "jumps"]            (length filter - all pass)
// Step 5: ["quick", "brown", "fox", "jump"]             (stemming)
//
// WHY THIS MATTERS:
// -----------------
// Proper analysis ensures:
// - "Running" matches "run", "runs", "ran"
// - "The dog" matches "DOG" (case insensitive)
// - Common words don't pollute the index
// - Search results are relevant and accurate
// ═══════════════════════════════════════════════════════════════════════════════

package iskra

import (
	"strings"
	"unicode"

	snowballeng "github.com/kljensen/snowball/english"
)

// AnalyzerConfig holds configuration options for text analysis
//
// This allows customization of the analysis pipeline without modifying code.
// Future enhancements could add language support, custom stopwords, etc.
type AnalyzerConfig struct {
	MinTokenLength  int  // Minimum token length to keep (default: 2)
	EnableStemming  bool // Whether to apply stemming (default: true)
	EnableStopwords bool // Whether to remove stopwords (default: true)
}

// DefaultConfig returns the standard analyzer configuration
func DefaultConfig() AnalyzerConfig {
	return AnalyzerConfig{
		MinTokenLength:  2,
		EnableStemming:  true,
		EnableStopwords: true,
	}
}

// Analyze transforms raw text into searchable tokens using the default pipeline
//
// This is the main entry point for text analysis. It applies all filters in sequence:
// 1. Tokenization
// 2. Lowercasing
// 3. Stopword filtering
// 4. Length filtering
// 5. Stemming
//
// Example:
//
//	tokens := Analyze("The quick brown fox jumps over the lazy dog")
//	// Returns: ["quick", "brown", "fox", "jump", "lazi", "dog"]
func Analyze(text string) []string {
	return AnalyzeWithConfig(text, DefaultConfig())
}

// AnalyzeWithConfig transforms text using a custom configuration
//
// This allows fine-grained control over the analysis pipeline.
//
// Example:
//
//	config := AnalyzerConfig{MinTokenLength: 3, EnableStemming: false}
//	tokens := AnalyzeWithConfig("The quick brown fox", config)
func AnalyzeWithConfig(text string, config AnalyzerConfig) []string {
	tokens := tokenize(text)
	tokens = lowercaseFilter(tokens)

	if config.EnableStopwords {
		tokens = stopwordFilter(tokens)
	}

	tokens = lengthFilter(tokens, config.MinTokenLength)

	if config.EnableStemming {
		tokens = stemmerFilter(tokens)
	}

	return tokens
}

// tokenize splits text into individual words
//
// ALGORITHM:
// ----------
// Uses Unicode-aware splitting: any non-letter and non-digit character is a delimiter.
//
// Examples:
//
//	"hello-world"      → ["hello", "world"]
//	"user@email.com"   → ["user", "email", "com"]
//	"price: $9.99"     → ["price", "9", "99"]
//	"café"             → ["café"]  (Unicode letters preserved)
//
// Why FieldsFunc?
// - Handles Unicode properly (unlike simple string splitting)
// - Treats multiple delimiters as one (no empty tokens)
// - Fast and memory efficient (Go standard library optimization)
func tokenize(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		// Split on any character that is not a letter or a number
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
}

// lowercaseFilter normalizes token casing
//
// WHY IT MATTERS:
// ---------------
// Without lowercasing, "Quick", "quick", and "QUICK" would be treated as
// different words, creating a poor search experience.
//
// Example:
//
//	["Hello", "World"] → ["hello", "world"]
//
// Performance Note:
// - Pre-allocates slice to avoid dynamic growth
// - Uses strings.ToLower for proper Unicode handling
func lowercaseFilter(tokens []string) []string {
	r := make([]string, len(tokens))
	for i, token := range tokens {
		r[i] = strings.ToLower(token)
	}
	return r
}

// stopwordFilter removes common English words that don't add search value
//
// STOPWORDS EXPLAINED:
// --------------------
// Words like "the", "a", "is" appear in almost every document, so they:
// - Waste index space
// - Don't help distinguish documents
// - Slow down search
//
// Example:
//
//	["the", "quick", "brown", "fox"] → ["quick", "brown", "fox"]
//
// Implementation Note:
// - Uses map lookup for O(1) checking
// - Pre-allocates capacity to reduce reallocations
func stopwordFilter(tokens []string) []string {
	r := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if !isStopword(token) {
			r = append(r, token)
		}
	}
	return r
}

// lengthFilter removes tokens that are too short to be meaningful
//
// WHY FILTER BY LENGTH?
// ---------------------
// Very short tokens (1-2 characters) are often:
// - Not semantically meaningful ("a", "i", "to")
// - Result in too many false matches
// - Already caught by stopword filter
//
// Example (minLength=2):
//
//	["a", "go", "cat", "i"] → ["go", "cat"]
//
// Performance:
// - O(n) single pass
// - Pre-allocated capacity
func lengthFilter(tokens []string, minLength int) []string {
	r := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if len(token) >= minLength {
			r = append(r, token)
		}
	}
	return r
}

// stemmerFilter reduces words to their root form
//
// STEMMING EXPLAINED:
// -------------------
// Stemming removes suffixes to find the word root:
//
//	"running", "runs", "ran" → "run"
//	"connection", "connected", "connecting" → "connect"
//
// WHY IT MATTERS:
// ---------------
// Without stemming, a search for "run" wouldn't match documents containing
// "running" or "runs", even though they're clearly related.
//
// ALGORITHM:
// ----------
// Uses the Snowball (Porter2) stemmer, which applies linguistic rules
// to remove common English suffixes.
//
// Example:
//
//	["running", "quickly", "foxes"] → ["run", "quick", "fox"]
//
// Trade-offs:
// + Improves recall (finds more relevant documents)
// + Reduces index size (fewer unique terms)
// - May over-stem (e.g., "university" → "univers")
// - Language-specific (this implementation is English-only)
func stemmerFilter(tokens []string) []string {
	r := make([]string, len(tokens))
	for i, token := range tokens {
		r[i] = snowballeng.Stem(token, false)
	}
	return r
}

// isStopword checks if a token is a common English stopword
//
// Uses a hash map for O(1) lookup performance.
// The map uses struct{} as values (0 bytes) instead of strings (16 bytes)
// for memory efficiency.
func isStopword(token string) bool {
	_, exists := englishStopwords[token]
	return exists
}

// englishStopwords contains common English words to exclude from indexing
//
// MEMORY OPTIMIZATION:
// --------------------
// Uses struct{} (empty struct) as the value type instead of string or bool.
// - struct{}: 0 bytes per entry
// - string:   16 bytes per entry
// - bool:     1 byte per entry
//
// For 300+ stopwords, this saves ~5KB of memory.
//
// STOPWORD SELECTION:
// -------------------
// This list includes:
// - Articles: a, an, the
// - Prepositions: in, on, at, to
// - Conjunctions: and, but, or
// - Pronouns: he, she, it, they
// - Common verbs: is, are, was, were
// - Numbers: one, two, three, etc.
var englishStopwords = map[string]struct{}{
	"a":            {},
	"about":        {},
	"above":        {},
	"across":       {},
	"after":        {},
	"afterwards":   {},
	"again":        {},
	"against":      {},
	"all":          {},
	"almost":       {},
	"alone":        {},
	"along":        {},
	"already":      {},
	"also":         {},
	"although":     {},
	"always":       {},
	"am":           {},
	"among":        {},
	"amongst":      {},
	"amoungst":     {},
	"amount":       {},
	"an":           {},
	"and":          {},
	"another":      {},
	"any":          {},
	"anyhow":       {},
	"anyone":       {},
	"anything":     {},
	"anyway":       {},
	"anywhere":     {},
	"are":          {},
	"around":       {},
	"as":           {},
	"at":           {},
	"back":         {},
	"be":           {},
	"became":       {},
	"because":      {},
	"become":       {},
	"becomes":      {},
	"becoming":     {},
	"been":         {},
	"before":       {},
	"beforehand":   {},
	"behind":       {},
	"being":        {},
	"below":        {},
	"beside":       {},
	"besides":      {},
	"between":      {},
	"beyond":       {},
	"bill":         {},
	"both":         {},
	"bottom":       {},
	"but":          {},
	"by":           {},
	"call":         {},
	"can":          {},
	"cannot":       {},
	"cant":         {},
	"co":           {},
	"con":          {},
	"could":        {},
	"couldnt":      {},
	"cry":          {},
	"de":           {},
	"describe":     {},
	"detail":       {},
	"do":           {},
	"done":         {},
	"down":         {},
	"due":          {},
	"during":       {},
	"each":         {},
	"eg":           {},
	"eight":        {},
	"either":       {},
	"eleven":       {},
	"else":         {},
	"elsewhere":    {},
	"empty":        {},
	"enough":       {},
	"etc":          {},
	"even":         {},
	"ever":         {},
	"every":        {},
	"everyone":     {},
	"everything":   {},
	"everywhere":   {},
	"except":       {},
	"few":          {},
	"fifteen":      {},
	"fify":         {},
	"fill":         {},
	"find":         {},
	"fire":         {},
	"first":        {},
	"five":         {},
	"for":          {},
	"former":       {},
	"formerly":     {},
	"forty":        {},
	"found":        {},
	"four":         {},
	"from":         {},
	"front":        {},
	"full":         {},
	"further":      {},
	"get":          {},
	"give":         {},
	"go":           {},
	"had":          {},
	"has":          {},
	"hasnt":        {},
	"have":         {},
	"he":           {},
	"hence":        {},
	"her":          {},
	"here":         {},
	"hereafter":    {},
	"hereby":       {},
	"herein":       {},
	"hereupon":     {},
	"hers":         {},
	"herself":      {},
	"him":          {},
	"himself":      {},
	"his":          {},
	"how":          {},
	"however":      {},
	"hundred":      {},
	"ie":           {},
	"if":           {},
	"in":           {},
	"inc":          {},
	"indeed":       {},
	"interest":     {},
	"into":         {},
	"is":           {},
	"it":           {},
	"its":          {},
	"itself":       {},
	"keep":         {},
	"last":         {},
	"latter":       {},
	"latterly":     {},
	"least":        {},
	"less":         {},
	"ltd":          {},
	"made":         {},
	"many":         {},
	"may":          {},
	"me":           {},
	"meanwhile":    {},
	"might":        {},
	"mill":         {},
	"mine":         {},
	"more":         {},
	"moreover":     {},
	"most":         {},
	"mostly":       {},
	"move":         {},
	"much":         {},
	"must":         {},
	"my":           {},
	"myself":       {},
	"name":         {},
	"namely":       {},
	"neither":      {},
	"never":        {},
	"nevertheless": {},
	"next":         {},
	"nine":         {},
	"no":           {},
	"nobody":       {},
	"none":         {},
	"noone":        {},
	"nor":          {},
	"not":          {},
	"nothing":      {},
	"now":          {},
	"nowhere":      {},
	"of":           {},
	"off":          {},
	"often":        {},
	"on":           {},
	"once":         {},
	"one":          {},
	"only":         {},
	"onto":         {},
	"or":           {},
	"other":        {},
	"others":       {},
	"otherwise":    {},
	"our":          {},
	"ours":         {},
	"ourselves":    {},
	"out":          {},
	"over":         {},
	"own":          {},
	"part":         {},
	"per":          {},
	"perhaps":      {},
	"please":       {},
	"put":          {},
	"rather":       {},
	"re":           {},
	"same":         {},
	"see":          {},
	"seem":         {},
	"seemed":       {},
	"seeming":      {},
	"seems":        {},
	"serious":      {},
	"several":      {},
	"she":          {},
	"should":       {},
	"show":         {},
	"side":         {},
	"since":        {},
	"sincere":      {},
	"six":          {},
	"sixty":        {},
	"so":           {},
	"some":         {},
	"somehow":      {},
	"someone":      {},
	"something":    {},
	"sometime":     {},
	"sometimes":    {},
	"somewhere":    {},
	"still":        {},
	"such":         {},
	"system":       {},
	"take":         {},
	"ten":          {},
	"than":         {},
	"that":         {},
	"the":          {},
	"their":        {},
	"them":         {},
	"themselves":   {},
	"then":         {},
	"thence":       {},
	"there":        {},
	"thereafter":   {},
	"thereby":      {},
	"therefore":    {},
	"therein":      {},
	"thereupon":    {},
	"these":        {},
	"they":         {},
	"thickv":       {},
	"thin":         {},
	"third":        {},
	"this":         {},
	"those":        {},
	"though":       {},
	"three":        {},
	"through":      {},
	"throughout":   {},
	"thru":         {},
	"thus":         {},
	"to":           {},
	"together":     {},
	"too":          {},
	"top":          {},
	"toward":       {},
	"towards":      {},
	"twelve":       {},
	"twenty":       {},
	"two":          {},
	"un":           {},
	"under":        {},
	"until":        {},
	"up":           {},
	"upon":         {},
	"us":           {},
	"very":         {},
	"via":          {},
	"was":          {},
	"we":           {},
	"well":         {},
	"were":         {},
	"what":         {},
	"whatever":     {},
	"when":         {},
	"whence":       {},
	"whenever":     {},
	"where":        {},
	"whereafter":   {},
	"whereas":      {},
	"whereby":      {},
	"wherein":      {},
	"whereupon":    {},
	"wherever":     {},
	"whether":      {},
	"which":        {},
	"while":        {},
	"whither":      {},
	"who":          {},
	"whoever":      {},
	"whole":        {},
	"whom":         {},
	"whose":        {},
	"why":          {},
	"will":         {},
	"with":         {},
	"within":       {},
	"without":      {},
	"would":        {},
	"yet":          {},
	"you":          {},
	"your":         {},
	"yours":        {},
	"yourself":     {},
	"yourselves":   {},
}
