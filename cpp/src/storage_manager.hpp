#pragma once

#include <string>
#include <vector>
#include <memory>
#include <rocksdb/db.h>
#include <rocksdb/options.h>

// Forward declarations
class RedisClient;
class GCSClient;

class StorageManager {
public:
    StorageManager();
    ~StorageManager();
    
    // Initialize all storage systems
    bool initialize();
    
    // Document operations
    bool storeDocument(int docId, const std::vector<float>& embedding, float temperature);
    bool retrieveDocument(int docId, std::vector<float>& embedding);
    bool deleteDocument(int docId);
    
    // Tier management
    bool moveToTier1(int docId, const std::vector<float>& embedding);
    bool moveToTier2(int docId, const std::vector<float>& embedding);
    bool moveToTier3(int docId, const std::vector<float>& embedding);
    
    // Temperature-based tier determination
    int determineTier(float temperature);
    
    // Statistics
    void printTierStats();
    
private:
    std::unique_ptr<RedisClient> redis_;
    rocksdb::DB* rocksdb_;
    std::unique_ptr<GCSClient> gcs_;
    
    // Tier thresholds
    float tier1_threshold_;
    float tier2_threshold_;
    
    // Helper methods
    bool storeInRedis(int docId, const std::vector<float>& embedding);
    bool storeInRocksDB(int docId, const std::vector<float>& embedding);
    bool storeInGCS(int docId, const std::vector<float>& embedding);
    
    bool retrieveFromRedis(int docId, std::vector<float>& embedding);
    bool retrieveFromRocksDB(int docId, std::vector<float>& embedding);
    bool retrieveFromGCS(int docId, std::vector<float>& embedding);
    
    bool deleteFromRedis(int docId);
    bool deleteFromRocksDB(int docId);
    bool deleteFromGCS(int docId);
};
