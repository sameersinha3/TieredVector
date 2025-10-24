#include <iostream>
#include <vector>
#include <random>
#include <signal.h>
#include <thread>
#include <chrono>
#include "storage_manager.hpp"
#include "http_server.hpp"

std::unique_ptr<HttpServer> http_server;
std::unique_ptr<StorageManager> storage;

void signalHandler(int signum) {
    std::cout << "\nShutting down server..." << std::endl;
    if (http_server) {
        http_server->stop();
    }
    exit(signum);
}

int main() {
    std::cout << "Starting 3-Tier Vector Storage System with HTTP API..." << std::endl;
    
    // Set up signal handler for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Initialize storage manager
    storage = std::make_unique<StorageManager>();
    if (!storage->initialize()) {
        std::cerr << "Failed to initialize storage system" << std::endl;
        return 1;
    }
    
    // Start HTTP server
    http_server = std::make_unique<HttpServer>(storage.get(), 8082);
    if (!http_server->start()) {
        std::cerr << "Failed to start HTTP server" << std::endl;
        return 1;
    }
    
    std::cout << "\nServer is running. Press Ctrl+C to stop." << std::endl;
    std::cout << "Endpoints:" << std::endl;
    std::cout << "  POST /store - Store a document" << std::endl;
    std::cout << "  GET /status - Get server status" << std::endl;
    
    // Keep server running
    while (http_server->isRunning()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    return 0;
}