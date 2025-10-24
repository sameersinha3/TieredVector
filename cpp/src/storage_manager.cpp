#include "storage_manager.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <hiredis/hiredis.h>

#ifdef HAVE_GCS
#include <google/cloud/storage/client.h>
#endif

class RedisClient {
public:
    RedisClient() : context_(nullptr) {}
    
    bool connect() {
        context_ = redisConnect("127.0.0.1", 6379);
        return context_ != nullptr && !context_->err;
    }
    
    bool set(const std::string& key, const std::string& value) {
        redisReply* reply = (redisReply*)redisCommand(context_, "SET %s %b", 
                                                     key.c_str(), value.data(), value.size());
        bool success = reply && reply->type == REDIS_REPLY_STATUS;
        freeReplyObject(reply);
        return success;
    }
    
    bool get(const std::string& key, std::string& value) {
        redisReply* reply = (redisReply*)redisCommand(context_, "GET %s", key.c_str());
        if (reply && reply->type == REDIS_REPLY_STRING) {
            value = std::string(reply->str, reply->len);
            freeReplyObject(reply);
            return true;
        }
        freeReplyObject(reply);
        return false;
    }
    
    bool del(const std::string& key) {
        redisReply* reply = (redisReply*)redisCommand(context_, "DEL %s", key.c_str());
        bool success = reply && reply->type == REDIS_REPLY_INTEGER;
        freeReplyObject(reply);
        return success;
    }
    
    ~RedisClient() {
        if (context_) {
            redisFree(context_);
        }
    }
    
private:
    redisContext* context_;
};

#ifdef HAVE_GCS
class GCSClient {
public:
    bool initialize() {
        try {
            client_ = google::cloud::storage::Client::CreateDefaultClient();
            return client_ != nullptr;
        } catch (...) {
            return false;
        }
    }
    
    bool store(const std::string& objectName, const std::string& data) {
        try {
            auto bucket = client_->Bucket("your-bucket-name");
            auto object = bucket.Object(objectName);
            auto writer = object.CreateWriter();
            writer.write(data.data(), data.size());
            writer.Close();
            return true;
        } catch (...) {
            return false;
        }
    }
    
    bool retrieve(const std::string& objectName, std::string& data) {
        try {
            auto bucket = client_->Bucket("your-bucket-name");
            auto object = bucket.Object(objectName);
            auto reader = object.CreateReader();
            data = std::string(std::istreambuf_iterator<char>(reader), {});
            return true;
        } catch (...) {
            return false;
        }
    }
    
    bool remove(const std::string& objectName) {
        try {
            auto bucket = client_->Bucket("your-bucket-name");
            auto object = bucket.Object(objectName);
            object.Delete();
            return true;
        } catch (...) {
            return false;
        }
    }
    
private:
    std::unique_ptr<google::cloud::storage::Client> client_;
};
#else
// Dummy GCS client when GCS is not available
class GCSClient {
public:
    bool initialize() { return false; }
    bool store(const std::string&, const std::string&) { return false; }
    bool retrieve(const std::string&, std::string&) { return false; }
    bool remove(const std::string&) { return false; }
};
#endif

StorageManager::StorageManager() 
    : redis_(std::make_unique<RedisClient>())
    , rocksdb_(nullptr)
    , gcs_(std::make_unique<GCSClient>())
    , tier1_threshold_(0.8f)
    , tier2_threshold_(0.5f) {
}

StorageManager::~StorageManager() {
    if (rocksdb_) {
        delete rocksdb_;
    }
}

bool StorageManager::initialize() {
    std::cout << "Initializing 3-tier storage system..." << std::endl;
    
    // Initialize Redis
    if (!redis_->connect()) {
        std::cerr << "Failed to connect to Redis" << std::endl;
        return false;
    }
    std::cout << "Redis connected" << std::endl;
    
    // Initialize RocksDB
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::Status status = rocksdb::DB::Open(options, "./rocksdb_data", &rocksdb_);
    if (!status.ok()) {
        std::cerr << "Failed to open RocksDB: " << status.ToString() << std::endl;
        return false;
    }
    std::cout << "RocksDB opened" << std::endl;
    
    // Initialize GCS (optional)
    if (gcs_->initialize()) {
        std::cout << "GCS client initialized" << std::endl;
    } else {
        std::cout << "GCS client not available (Tier 3 disabled)" << std::endl;
    }
    
    return true;
}

int StorageManager::determineTier(float temperature) {
    if (temperature >= tier1_threshold_) {
        return 1; // Redis
    } else if (temperature >= tier2_threshold_) {
        return 2; // RocksDB
    } else {
        return 3; // GCS
    }
}

bool StorageManager::storeDocument(int docId, const std::vector<float>& embedding, float temperature) {
    int tier = determineTier(temperature);
    
    bool success = false;
    switch (tier) {
        case 1:
            success = storeInRedis(docId, embedding);
            break;
        case 2:
            success = storeInRocksDB(docId, embedding);
            break;
        case 3:
            // If GCS is not available, store in RocksDB instead
            success = storeInRocksDB(docId, embedding);
            break;
        default:
            success = false;
    }
    
    if (!success) {
        std::cerr << "Failed to store vector " << docId << " in tier " << tier << std::endl;
    }
    
    return success;
}

bool StorageManager::retrieveDocument(int docId, std::vector<float>& embedding) {
    // Try Tier 1 (Redis) first
    if (retrieveFromRedis(docId, embedding)) {
        return true;
    }
    
    // Try Tier 2 (RocksDB)
    if (retrieveFromRocksDB(docId, embedding)) {
        return true;
    }
    
    // Try Tier 3 (GCS)
    if (retrieveFromGCS(docId, embedding)) {
        return true;
    }
    
    return false;
}

bool StorageManager::storeInRedis(int docId, const std::vector<float>& embedding) {
    std::string key = "vector:" + std::to_string(docId);
    std::string value(reinterpret_cast<const char*>(embedding.data()), 
                     embedding.size() * sizeof(float));
    return redis_->set(key, value);
}

bool StorageManager::storeInRocksDB(int docId, const std::vector<float>& embedding) {
    std::string key = "vector_" + std::to_string(docId);
    std::string value(reinterpret_cast<const char*>(embedding.data()), 
                     embedding.size() * sizeof(float));
    rocksdb::Status status = rocksdb_->Put(rocksdb::WriteOptions(), key, value);
    return status.ok();
}

bool StorageManager::storeInGCS(int docId, const std::vector<float>& embedding) {
    std::string objectName = "vector_" + std::to_string(docId);
    std::string value(reinterpret_cast<const char*>(embedding.data()), 
                     embedding.size() * sizeof(float));
    return gcs_->store(objectName, value);
}

bool StorageManager::retrieveFromRedis(int docId, std::vector<float>& embedding) {
    std::string key = "vector:" + std::to_string(docId);
    std::string value;
    if (redis_->get(key, value)) {
        embedding.resize(value.size() / sizeof(float));
        std::memcpy(embedding.data(), value.data(), value.size());
        return true;
    }
    return false;
}

bool StorageManager::retrieveFromRocksDB(int docId, std::vector<float>& embedding) {
    std::string key = "vector_" + std::to_string(docId);
    std::string value;
    rocksdb::Status status = rocksdb_->Get(rocksdb::ReadOptions(), key, &value);
    if (status.ok()) {
        embedding.resize(value.size() / sizeof(float));
        std::memcpy(embedding.data(), value.data(), value.size());
        return true;
    }
    return false;
}

bool StorageManager::retrieveFromGCS(int docId, std::vector<float>& embedding) {
    std::string objectName = "vector_" + std::to_string(docId);
    std::string value;
    if (gcs_->retrieve(objectName, value)) {
        embedding.resize(value.size() / sizeof(float));
        std::memcpy(embedding.data(), value.data(), value.size());
        return true;
    }
    return false;
}

void StorageManager::printTierStats() {
    std::cout << "\n=== Tier Statistics ===" << std::endl;
    std::cout << "Tier 1 (Redis) threshold: " << tier1_threshold_ << std::endl;
    std::cout << "Tier 2 (RocksDB) threshold: " << tier2_threshold_ << std::endl;
    std::cout << "Tier 3 (GCS) threshold: < " << tier2_threshold_ << std::endl;
}
