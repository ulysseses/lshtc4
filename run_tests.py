import unittest
import test.all_tests

'''
unittest boilerplate code

With this setup, you can indeed just `import module` in your test modules.
The downside is you would need more support code to execute a particular
test... I just run them all everytime.
'''

testSuite =  test.all_tests.create_test_suite()
text_runner = unittest.TextTestRunner().run(testSuite)