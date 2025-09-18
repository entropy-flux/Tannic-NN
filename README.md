# Tannic-NN

Tannic-NN is a lightweight, modular C++ neural network inference engine built on top of the [Tannic](https://github.com/entropy-flux/Tannic) tensor library. It provides a modular way to construct and run neural network models in pure C++. 

You can find examples of neural networks inference with the tannic framework here: 
- [CNN Server Example](https://github.com/entropy-flux/cnn-server-example) – demonstrates serving convolutional neural network models.  
- [ViT Server Example](https://github.com/entropy-flux/vit-server-example) – demonstrates serving Vision Transformer models.  
- [LLaMa3 Text Completion Server Example](https://github.com/entropy-flux/llama3-server-example) – demonstrates serving a transformer server for text completion tasks.

### Getting Started

Define your models as you would do it with pytorch.

```cpp
#include <tannic.hpp>
#include <tannic-nn.hpp>

using namespace tannic;

struct MyModel : nn::Module {
    nn::Linear fc1;
    nn::Linear fc2;

    constexpr MyModel(type dtype, int input_dim, int hidden_dim, int output_dim)
    :   fc1(dtype, input_dim, hidden_dim,  /*bias=*/true)
    ,   fc2(dtype, hidden_dim, output_dim, /*bias=*/false) {}

    Tensor forward(Tensor features) const {
        features = fc1(features);
        return fc2(nn::relu(features));
    }

    void initialize(nn::Parameters& params) const {
        fc1.initialize("fc1", params);
        fc2.initialize("fc2", params);
    }
};
```

The only difference is that you must define an initialize method in your module.
This method links the module’s parameters to an external nn::Parameters object, allowing you to load pretrained weights, similar to how frameworks like llama.cpp handle weight loading.

Unlike other frameworks that store everything in a single file, Tannic-NN separates the model into two parts:

- Metadata file: describes the model architecture and how to interpret the raw weights, and can be easily shared with your model's code, since it will be just a few kb. 

- Weights file: a single contiguous buffer containing all raw model data and can be as big as you want. 

When a model is initialized, all parameters are mapped into a single big contiguous buffer. Each parameter does not own its own memory but instead references a slice of the buffer using an offset and shape descriptor. During inference, operations only need the pointer, offset, and shape, avoiding memory duplication and keeping execution memory-efficient.


### Installation

#### Requirements

- C++23 (tested with GCC ≥ 14, Clang ≥ 18)

- CMake ≥ 3.28
 
 #### Building from source

Clone the repository and fetch tannic tensor library:

```bash
git clone https://github.com/entropy-flux/Tannic-NN
cd Tannic
git submodule update --init --recursive
```

Then build the library:

```bash
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
ctest --output-on-failure 
```