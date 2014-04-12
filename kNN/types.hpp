#ifndef UTILS_H
#define UTILS_H

typedef float flt;
typedef size_t DOC;
typedef size_t WORD;
typedef size_t LABEL;
typedef size_t CAT;
typedef size_t PCAT;

struct Word {
	size_t doc, word
	flt tfidf
};

struct Label {
	size_t doc, label
};

struct Doc {
	size_t word, doc
};

#endif