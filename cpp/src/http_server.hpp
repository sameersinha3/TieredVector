#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>

class StorageManager;

class HttpServer {
public:
    HttpServer(StorageManager* storage, int port = 8080);
    ~HttpServer();
    
    bool start();
    void stop();
    bool isRunning() const;
    
private:
    StorageManager* storage_;
    int port_;
    std::atomic<bool> running_;
    std::unique_ptr<std::thread> server_thread_;
    
    void runServer();
    std::string handleRequest(const std::string& method, const std::string& path, const std::string& body);
    std::string handleStoreRequest(const std::string& body);
    std::string handleRetrieveRequest(const std::string& body);
    std::string handleStatusRequest();
};
