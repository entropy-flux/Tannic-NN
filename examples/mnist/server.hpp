#include <tannic.hpp>
#include <tannic-nn.hpp>
 
#include <unistd.h>
#include <netinet/in.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdint>
#include <arpa/inet.h>  

using namespace tannic;

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

    Metadata(Tensor const& tensor) {
        offset = tensor.offset();
        nbytes = tensor.nbytes();
        dtype = tensor.dtype();
        rank = tensor.rank();
        shape = tensor.shape().address();
        strides = tensor.strides().address();
    }   
};

std::size_t msizeof(Metadata const& metadata) { 
    return sizeof(metadata.offset) + sizeof(metadata.nbytes) + sizeof(uint8_t) + 
           sizeof(metadata.rank) + (sizeof(std::size_t) * metadata.rank) * 2;
}

Tensor deserialize(std::vector<std::byte> const& metadata) { 
    Metadata structured(metadata);
    Shape shape(structured.shape, structured.shape + structured.rank);
    Strides strides(structured.strides, structured.strides + structured.rank);
    std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(structured.nbytes);
    return Tensor(structured.dtype, shape, strides, structured.offset, buffer);
} 

std::vector<std::byte> serialize(Tensor const& tensor) {
    Metadata structured(tensor);
    uint8_t msize = msizeof(structured);
    std::array<std::byte, 5> header = {std::byte('M'), std::byte('L'), std::byte('B'), std::byte('C'), static_cast<std::byte>(msize)}; 
    std::vector<std::byte> serialized(5 + msize + tensor.nbytes()); 
    std::memcpy(serialized.data(), header.data(), 5); std::byte* data = serialized.data() + 5;
    std::memcpy(data, &structured.offset, sizeof(std::size_t)); data += sizeof(std::size_t);
    std::memcpy(data, &structured.nbytes, sizeof(std::size_t)); data += sizeof(std::size_t);
    uint8_t dcode = dcodeof(structured.dtype);
    std::memcpy(data, &dcode, sizeof(uint8_t)); data += sizeof(uint8_t);
    std::memcpy(data, &structured.rank, sizeof(uint8_t)); data += sizeof(uint8_t);
    std::memcpy(data, structured.shape, sizeof(std::size_t) * structured.rank); data += sizeof(std::size_t) * structured.rank;
    std::memcpy(data, structured.strides, sizeof(std::size_t) * structured.rank); data += sizeof(std::size_t) * structured.rank;
    std::memcpy(data, tensor.bytes(), tensor.nbytes());
    return serialized; 
}


class Endpoint {
public:
    Endpoint() {
        std::memset(&storage_, 0, sizeof(storage_));
        addr4_.sin_family = AF_INET;
        addr4_.sin_addr.s_addr = INADDR_ANY;
        addr4_.sin_port = htons(0);
        length_ = sizeof(sockaddr_in);
    }

    Endpoint(uint16_t port, in_addr_t address = INADDR_ANY) {
        std::memset(&storage_, 0, sizeof(storage_));
        addr4_.sin_family = AF_INET;
        addr4_.sin_addr.s_addr = address;
        addr4_.sin_port = htons(port);
        length_ = sizeof(sockaddr_in);
    }

    sockaddr* address() {
        return reinterpret_cast<sockaddr*>(&storage_);
    }

    socklen_t* length() {
        return &length_;
    }

private:
    union {
        sockaddr_storage storage_;
        sockaddr_in addr4_;
        sockaddr_in6 addr6_;
    };
    socklen_t length_{};
};


class Socket {
public:
    Socket() {
        descriptor_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (descriptor_ < 0)
            throw std::runtime_error("socket() failed");
    }

    Socket(Socket&& other) noexcept : descriptor_(other.descriptor_) {
        other.descriptor_ = -1;
    }

    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            close();
            descriptor_ = other.descriptor_;
            other.descriptor_ = -1;
        }
        return *this;
    }

    ~Socket() { close(); }

    void reuse() {
        int option = 1;
        auto status = setsockopt(descriptor_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &option, sizeof(option));
        if (status < 0)
            throw std::runtime_error("setsockopt() failed");
    }

    void bind(uint16_t port, in_addr_t address = INADDR_ANY) {
        Endpoint endpoint(port, address);
        auto status = ::bind(descriptor_, endpoint.address(), *endpoint.length());
        if (status < 0)
            throw std::runtime_error("bind() failed");
    }

    void listen(int backlog = 3) {
        if (::listen(descriptor_, backlog) < 0)
            throw std::runtime_error("listen() failed");
    }

    Socket accept() {
        Endpoint client;
        int descriptor = ::accept(descriptor_, client.address(), client.length());
        if (descriptor < 0)
            throw std::runtime_error("accept() failed");
        return Socket(descriptor);
    }

    ssize_t send(const void* buffer, size_t length, int flags = 0) {
        return ::send(descriptor_, buffer, length, flags);
    }

    ssize_t receive(void* buffer, size_t length, int flags = 0) {
        return ::recv(descriptor_, buffer, length, flags);
    }

private:
    explicit Socket(int descriptor) : descriptor_(descriptor) {}

    void close() {
        if (descriptor_ >= 0) {
            ::close(descriptor_);
            descriptor_ = -1;
        }
    }

    int descriptor_{-1};
};
 
class Server { 

public:
    Server(int port) : listener_(), port_(port) {
        listener_.reuse();
        listener_.bind(port_);
        listener_.listen();
        std::cout << "Listening on port " << port_ << "...\n";
    }

    void run() {
        while (true) {
            auto client = listener_.accept(); 
            try { 
                std::array<std::byte, 5> header;  
                std::array<std::byte, 4> magic =  {std::byte('M'), std::byte('L'), std::byte('B'), std::byte('C') }; 

                read(client, header.data(), 5); 
 
                if (!std::equal(header.begin(), header.begin() + 4, magic.begin())) {
                    std::cerr << "Invalid magic header\n";
                    return;
                } 

                uint8_t msize = static_cast<uint8_t>(header[4]);
                std::vector<std::byte> metadata(msize);
                read(client, metadata.data(), msize);

                Tensor request = deserialize(metadata);
                read(client, request.bytes(), request.nbytes());

                Tensor response = forward(request);  
                std::vector<std::byte> serialized = serialize(response); 
                client.send(serialized.data(), serialized.size());

            } catch (const std::exception& exception) {
                std::cerr << "Error: " << exception.what() << "\n";
            }
        }
    }

    Tensor forward(Tensor input) const;

private:
    int port_;
    Socket listener_;

    static void read(Socket& socket, void* buffer, size_t length) {
        size_t total = 0;
        while (total < length) {
            ssize_t n = socket.receive(static_cast<char*>(buffer) + total, length - total);
            if (n <= 0)
                throw std::runtime_error("Socket closed or error while reading");
            total += n;
        }
    } 
};

