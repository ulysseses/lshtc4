#include <vector>
#include <iostream>

#include "module_api.h"

using namespace std;

int main() {
    import_module();
    vector<int> *v = func();
    //cout << "v[0] = " << v[0] << endl;
}