#include <iostream>
#include <asio.hpp>
#include <thread>
#include <vector>
#include <array>
#include <cstdint>
#include <tannic.hpp>
#include <tannic-nn.hpp>

using asio::ip::tcp;
using asio::io_context;

using namespace tannic;

constexpr std::array<std::byte, 4> magic{ std::byte('M'), std::byte('L'), std::byte('B'), std::byte('C')};  
 
struct Metadata {
    std::size_t offset;
    std::size_t nbytes;
    type dtype;
    uint8_t rank;
    std::size_t const* shape;
    std::size_t const* strides; 

    Metadata(std::vector<std::byte> const& bytes) {
        std::byte const* data = bytes.data();  
        offset = *reinterpret_cast<const std::size_t*>(data); data += sizeof(std::size_t);
        nbytes = *reinterpret_cast<const std::size_t*>(data); data += sizeof(std::size_t);
        dtype = dtypeof(*reinterpret_cast<const uint8_t*>(data)); data += sizeof(uint8_t);
        rank = *reinterpret_cast<const uint8_t*>(data); data += sizeof(uint8_t);
        shape = reinterpret_cast<const std::size_t*>(data); data += sizeof(std::size_t) * rank;
        strides = reinterpret_cast<const std::size_t*>(data); 
    }
};

Tensor deserialize(std::vector<std::byte> const& metadata) { 
    Metadata structured(metadata);
    Shape shape(structured.shape, structured.shape + structured.rank);
    Strides strides(structured.strides, structured.strides + structured.rank);
    std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(structured.nbytes);
    return Tensor(structured.dtype, shape, strides, structured.offset, buffer);
}


struct MLP : nn::Module {
    nn::Linear input_layer;
    nn::Linear output_layer;

    constexpr MLP(type dtype, size_t input_features, size_t hidden_features, size_t output_features) 
    :   input_layer(dtype, input_features, hidden_features)
    ,   output_layer(dtype, hidden_features, output_features)
    {}

    Tensor forward(Tensor features) const {  
        features = nn::relu(input_layer(features));
        return output_layer(features); 
    }

    void initialize() const {
        input_layer.initialize("input_layer");
        output_layer.initialize("output_layer");
    }
}; 

constexpr MLP model(float32, 784, 256, 10); 

class Server {
public:
    explicit Server(unsigned short port);
    void run();

private:
    void handle_client(tcp::socket socket) {
        try {   
            std::array<std::byte, 5> header;
            asio::read(socket, asio::buffer(header)); 
            
            if (!std::equal(header.begin(), header.begin() + 4, magic.begin())) {
                std::cerr << "Invalid magic header\n";
                return;
            }
 
            auto msize = static_cast<uint8_t>(header[4]);
            if (msize == 0 || msize > 255) {  
                std::cerr << "Invalid metadata size\n";
                return;
            }
 
            std::vector<std::byte> metadata(msize);
            asio::read(socket, asio::buffer(metadata)); 

            Tensor request = deserialize(metadata); 
            asio::read(socket, asio::buffer(request.bytes(), request.nbytes()));  


            Tensor response = model(request);
            std::cout << response;
            
            throw std::runtime_error("OK!");
 /*
            std::string response = "OK"; 
            uint64_t resp_len = response.size();
            std::array<char, 8> resp_len_buf{};
            std::memcpy(resp_len_buf.data(), &resp_len, sizeof(resp_len));
            asio::write(socket, asio::buffer(resp_len_buf)); 
            asio::write(socket, asio::buffer(response)); */
        } catch (const std::exception& exception) {
            std::cerr << "Client error: " << exception.what() << std::endl;
        }
    }

    unsigned short port_; 
    io_context context_;
    tcp::acceptor acceptor_;
}; 
 

Server::Server(unsigned short port)
:   port_(port) 
,   acceptor_(context_, tcp::endpoint(tcp::v4(), port)) {} 


void Server::run() {
    std::cout << "Server listening on port " << acceptor_.local_endpoint().port() << "\n";

    for (;;) {
        tcp::socket socket(context_);
        acceptor_.accept(socket);
        std::thread(&Server::handle_client, this, std::move(socket)).detach();
    }
} 
  

int main() {
    nn::Parameters::initialize("./data/mlp");
    try {
        Server server(8080);
        model.initialize();
        server.run();
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
    return 0;
}