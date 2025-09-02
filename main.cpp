#include <iostream>

struct Test {
    template <typename Self>
    void greet(this Self&& self) {
        
    }
};

int main() {
    Test test;
    test.greet();
    return 0;
}