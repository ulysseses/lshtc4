import glob
import unittest

'''
unittest boilerplate code

With this setup, you can indeed just `import module` in your test modules.
The downside is you would need more support code to execute a particular
test... I just run them all everytime.
'''

def create_test_suite():
	# list of strings that follow glob pattern
	test_file_strings = glob.glob('test/test_*.py')
	# 'test.' is package (test folder), 5 ignores 'test.',
	# and len(fname)-3 ignores the '.py' extension
	module_strings = ['test.' + fname[5:len(fname)-3] \
		for fname in test_file_strings]
	# a list of suites
	suites = [unittest.defaultTestLoader.loadTestsFromName(name) \
		for name in module_strings]
	# testSuite is a class container of a list of suites
	# i.e. a list of suites converted into another suite
	# aka testSuite is the union of all its suites
	testSuite = unittest.TestSuite(suites)
	return testSuite
