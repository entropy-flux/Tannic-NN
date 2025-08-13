#include <iostream>
#include <memory>

int main() {
    std::shared_ptr<int> sp = std::make_shared<int>(42);
 
    void* ptr = (void*)(&sp);
 
    std::shared_ptr<int> sp_ptr = *(std::shared_ptr<int>*)(ptr);

    std::cout << sp_ptr.use_count();
 
    std::cout << "Value: " << *sp_ptr << "\n"; // prints 42

    return 0;
}
