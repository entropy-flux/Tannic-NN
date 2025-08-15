#ifndef EMBEDDINGS_HPP
#define EMBEDDINGS_HPP

#include <tannic.hpp>  
#include <tannic/Slices.hpp>
#include "Parameters.hpp"
#include "Modules.hpp"

using tannic::expression::Slice;

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
        Tensor result(dtype(), Shape(sizeof...(indexes), weight_.shape().back())); 
        forward(result, lookup);
        return result;
    }

    Tensor forward(Tensor const& indexes) const { 
        assert(indexes.rank() == 1 && "Indexes should be a vector"); 
        Tensor result(dtype(), Shape(indexes.shape().front(), weight_.shape().back())); 
        forward(result, indexes);
        return result;
    }

    constexpr type dtype() const { 
        return weight_.dtype(); 
    } 

    Parameter const& weight() const { 
        return weight_; 
    }

protected:
    void forward(Tensor& result, std::vector<int64_t> const& lookup) const; 
    void forward(Tensor& result, Tensor const& lookup) const;
 

public: 
    template<Integral Index>
    constexpr auto operator[](Index index) const {    
        return Slice<Parameter, Index>(weight_, std::make_tuple(index));
    }

    constexpr auto operator[](indexing::Range range) const {  
        return Slice<Parameter, indexing::Range>(weight_, std::make_tuple(range));
    } 

    template<class ... Indexes>
    constexpr auto operator[](Indexes... indexes) const { 
        return Slice<Parameter, Indexes...>(weight_, std::make_tuple(indexes...));
    }  


private:  
    Parameter weight_; 
    mutable Allocator allocator_ = Host{};
};

} // namespace tannic

#endif