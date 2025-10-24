#include "http_server.hpp"
#include "storage_manager.hpp"
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

HttpServer::HttpServer(StorageManager* storage, int port) 
    : storage_(storage), port_(port), running_(false) {
}

HttpServer::~HttpServer() {
    stop();
}

bool HttpServer::start() {
    if (running_) {
        return true;
    }
    
    running_ = true;
    server_thread_ = std::make_unique<std::thread>(&HttpServer::runServer, this);
    
    // Give server time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    return true;
}

void HttpServer::stop() {
    if (running_) {
        running_ = false;
        if (server_thread_ && server_thread_->joinable()) {
            server_thread_->join();
        }
    }
}

bool HttpServer::isRunning() const {
    return running_;
}

void HttpServer::runServer() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    
    // Create socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "Socket creation failed" << std::endl;
        return;
    }
    
    // Set socket options
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        std::cerr << "Setsockopt failed" << std::endl;
        return;
    }
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);
    
    // Bind socket
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        return;
    }
    
    // Listen
    if (listen(server_fd, 3) < 0) {
        std::cerr << "Listen failed" << std::endl;
        return;
    }
    
    std::cout << "HTTP server listening on port " << port_ << std::endl;
    
    while (running_) {
        // Accept connection
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
            if (running_) {
                std::cerr << "Accept failed" << std::endl;
            }
            continue;
        }
        
        // Read request
        char buffer[4096] = {0};
        read(new_socket, buffer, 4096);
        
        std::string request(buffer);
        
        // Parse request
        std::istringstream request_stream(request);
        std::string method, path, version;
        request_stream >> method >> path >> version;
        
        // Find body
        std::string body;
        size_t body_start = request.find("\r\n\r\n");
        if (body_start != std::string::npos) {
            body = request.substr(body_start + 4);
        }
        
        // Handle request
        std::string response = handleRequest(method, path, body);
        
        // Send response
        send(new_socket, response.c_str(), response.length(), 0);
        close(new_socket);
    }
    
    close(server_fd);
}

std::string HttpServer::handleRequest(const std::string& method, const std::string& path, const std::string& body) {
    if (method == "POST" && path == "/store") {
        return handleStoreRequest(body);
    } else if (method == "POST" && path == "/retrieve") {
        return handleRetrieveRequest(body);
    } else if (method == "GET" && path == "/status") {
        return handleStatusRequest();
    } else {
        return "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n{\"error\":\"Not found\"}";
    }
}

std::string HttpServer::handleStoreRequest(const std::string& body) {
    try {
        // Simple JSON parsing (you might want to use a proper JSON library)
        // For now, we'll assume the format is correct
        
        // Extract doc_id, embedding, and temperature from JSON
        // This is a simplified parser - in production, use a proper JSON library
        
        size_t doc_id_start = body.find("\"doc_id\":") + 9;
        size_t doc_id_end = body.find(",", doc_id_start);
        int doc_id = std::stoi(body.substr(doc_id_start, doc_id_end - doc_id_start));
        
        size_t temp_start = body.find("\"temperature\":") + 14;
        size_t temp_end = body.find("}", temp_start);
        float temperature = std::stof(body.substr(temp_start, temp_end - temp_start));
        
        // Extract embedding array (simplified - assumes 768 dimensions)
        size_t embedding_start = body.find("\"embedding\":[") + 13;
        size_t embedding_end = body.find("]", embedding_start);
        std::string embedding_str = body.substr(embedding_start, embedding_end - embedding_start);
        
        // Parse embedding values
        std::vector<float> embedding;
        std::istringstream embedding_stream(embedding_str);
        std::string value;
        while (std::getline(embedding_stream, value, ',')) {
            embedding.push_back(std::stof(value));
        }
        
        // Store in storage manager
        bool success = storage_->storeDocument(doc_id, embedding, temperature);
        
        if (success) {
            return "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"success\"}";
        } else {
            return "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n\r\n{\"error\":\"Storage failed\"}";
        }
        
    } catch (const std::exception& e) {
        return "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n\r\n{\"error\":\"Invalid request\"}";
    }
}

std::string HttpServer::handleRetrieveRequest(const std::string& body) {
    // TODO: Implement retrieve endpoint
    return "HTTP/1.1 501 Not Implemented\r\nContent-Type: application/json\r\n\r\n{\"error\":\"Not implemented\"}";
}

std::string HttpServer::handleStatusRequest() {
    return "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"status\":\"running\"}";
}
