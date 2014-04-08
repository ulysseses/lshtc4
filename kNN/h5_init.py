import tables as tb

class XExample(tb.IsDescription):
	doc = tb.UInt32Col()
	word = tb.UInt32Col()
	count = tb.UInt32Col()

class ModdedXExample(tb.IsDescription):
	doc = tb.UInt32Col()
	word = tb.UInt32Col()
	tfidf = tb.Float32Col()

class YExample(tb.IsDescription):
	doc = tb.UInt32Col()
	label = tb.UInt32Col()

def initialize_h5(h5name="../working/train.h5", expectedrows=2600000):
	# Create a PyTables file from scratch
	h5file = tb.openFile(h5name, mode = "w", title = "Training Set")

	# Create a table for all Xs and Ys
	h5file.createTable(h5file.root, 'X', XExample, "raw X", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'Y', YExample, "raw Y", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'pruned1X', XExample, "label-pruned X", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'pruned1Y', YExample, "label-pruned Y", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'pruned2X', XExample, "word-pruned X", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'pruned2Y', YExample, "word-pruned Y", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'pruned3X', XExample, "bin-word-pruned X", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'pruned3Y', YExample, "bin-word-pruned Y", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'tfidfX', ModdedXExample, "tfidf X", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'vX', ModdedXExample, "validation X", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'vY', YExample, "validation Y", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'tX', ModdedXExample, "training X", expectedrows=expectedrows);
	h5file.createTable(h5file.root, 'tY', YExample, "training Y", expectedrows=expectedrows);

	h5file.close()