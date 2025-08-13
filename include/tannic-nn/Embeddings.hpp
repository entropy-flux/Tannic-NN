#ifndef EMBEDDINGS_HPP
#define EMBEDDINGS_HPP

#include "Parameters.hpp"
#include "Modules.hpp"

namespace tannic::nn {

class Embedding : public Module {
    public: 

    constexpr Embedding(type dtype, size_t lenght, size_t dimension) 
    :   weight_(dtype, Shape(lenght, dimension)) 
    {}

    void initialize(std::string const& name, Parameters& parameters) const { 
        weight_.initialize(name, parameters);
        allocator_ = parameters.allocator();
    }

    template<class... Indexes>
    Tensor forward(Indexes... indexes) const {     
        std::vector<int64_t> lookup{ static_cast<int64_t>(indexes)... }; 
        Tensor result(dtype(), Shape(sizeof...(indexes), weight_.shape().back()), weight_.strides()); 
        forward(result, lookup);
        return result;
    }

    constexpr type dtype() const { 
        return weight_.dtype(); 
    } 

    /*
    protected:
    void forward(Tensor& result, std::vector<int64_t> const& lookup) const;
    */

    private: 
    type itype_; 
    Parameter weight_; 
    mutable Allocator allocator_ = Host{};
};

} // namespace tannic

#endif