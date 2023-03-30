#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>

class Test {
public:
    int* data;
    Test(int* d) : data(d) {};
    ~Test() {};
};