#include <cstdint>
#include <iostream>
#include <vector>
#include <fstream>
#include "utils.hpp"
// #include "hinter_api.h"
#include <Python.h>
#include "hinter.h"

using namespace std;

/*
preproc
	doc2cat = vector<vector<CAT>>
	parents = vector<vector<PCAT>>
	children = vector<vector<CAT>>
	X_lens = vector<size_t>
	X_starts = vector<size_t>
	Y_lens = vector<size_t>
	Y_starts = vector<size_t>
	word2doc = vector<vector<DOC>>
*/

vector<vector<PCAT>> *parents_func(char *filename) {
	vector<vector<PCAT>> *parents = new vector<vector<PCAT>>{};
	ifstream hierarchy_file(filename, ifstream::in);
	CAT child;
	PCAT parent;
	while (hierarchy_file >> parent >> child) {
		if (child >= parents->size())
			parents->resize(child+1, vector<PCAT>());
		(*parents)[child].push_back(parent);
	}
	return parents;
}

vector<vector<CAT>> *children_func(vector<vector<PCAT>> *parents) {
	vector<vector<CAT>> *children = new vector<vector<CAT>>{};
	vector<PCAT> *inner;
	auto end = parents->end();
	CAT child = 0;
	PCAT parent;
	for (auto it = parents->begin(); it != end; ++it) {
		inner = &(*it); // vector of PCATs
		auto end2 = inner->end();
		for (auto it2 = inner->begin(); it2 != end2; ++it2) {
			parent = *it2;
			if (parent >= children->size())
				children->resize(parent+1, vector<CAT>());
			(*children)[parent].push_back(child);
		}
		child++;
	}
	return children;
}

int main() {
	// import_kNN__hinter();
    Py_Initialize();
    inithinter();
    Py_Finalize();

	cout << "hi" << endl;
	return 0;
}