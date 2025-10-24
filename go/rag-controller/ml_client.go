package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// MLClient handles communication with the Python ML service
type MLClient struct {
	baseURL    string
	httpClient *http.Client
}

// MLSearchRequest represents a request to the ML service
type MLSearchRequest struct {
	Query string `json:"query"`
	K     int    `json:"k"`
}

// MLSearchResponse represents a response from the ML service
type MLSearchResponse struct {
	QueryText        string    `json:"query_text"`
	RetrievedDocs    []int     `json:"retrieved_docs"`
	Similarities     []float64 `json:"similarities"`
	QueryEmbedding   []float64 `json:"query_embedding"`
}

// MLSimilarRequest represents a request for similar documents
type MLSimilarRequest struct {
	DocIdx    int `json:"doc_idx"`
	NSimilar  int `json:"n_similar"`
}

// MLSimilarResponse represents a response for similar documents
type MLSimilarResponse struct {
	DocIdx      int   `json:"doc_idx"`
	SimilarDocs []int `json:"similar_docs"`
	Count       int   `json:"count"`
}

// NewMLClient creates a new ML client
func NewMLClient(baseURL string) *MLClient {
	return &MLClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Search performs similarity search using the ML service
func (c *MLClient) Search(query string, k int) (*MLSearchResponse, error) {
	request := MLSearchRequest{
		Query: query,
		K:     k,
	}
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}
	
	resp, err := c.httpClient.Post(
		c.baseURL+"/search",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service error: %s", string(body))
	}
	
	var response MLSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	
	return &response, nil
}

// GetSimilarDocuments gets similar documents for a given document
func (c *MLClient) GetSimilarDocuments(docIdx int, nSimilar int) ([]int, error) {
	request := MLSimilarRequest{
		DocIdx:   docIdx,
		NSimilar: nSimilar,
	}
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}
	
	resp, err := c.httpClient.Post(
		c.baseURL+"/similar",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service error: %s", string(body))
	}
	
	var response MLSimilarResponse
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	
	return response.SimilarDocs, nil
}

// EncodeQuery encodes a query text into an embedding
func (c *MLClient) EncodeQuery(query string) ([]float64, error) {
	request := map[string]string{
		"query": query,
	}
	
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %v", err)
	}
	
	resp, err := c.httpClient.Post(
		c.baseURL+"/encode",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ML service error: %s", string(body))
	}
	
	var response struct {
		Query     string    `json:"query"`
		Embedding []float64 `json:"embedding"`
		Dimension int       `json:"dimension"`
	}
	
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	
	return response.Embedding, nil
}

// HealthCheck checks if the ML service is healthy
func (c *MLClient) HealthCheck() error {
	resp, err := c.httpClient.Get(c.baseURL + "/health")
	if err != nil {
		return fmt.Errorf("failed to make request: %v", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ML service is not healthy: status %d", resp.StatusCode)
	}
	
	return nil
}
