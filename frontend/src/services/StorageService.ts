/**
 * StorageService - Unified persistence layer using localStorage and IndexedDB
 *
 * localStorage: Simple key-value storage for lightweight data
 * IndexedDB: Object storage for complex data structures
 */

// IndexedDB configuration
const DB_NAME = 'celestial_studio';
const DB_VERSION = 1;
const STORES = {
  ALGORITHMS: 'algorithms',
  SCENARIOS: 'scenarios',
  PARAMETERS: 'parameters'
};

// localStorage keys
const LOCAL_KEYS = {
  CHAT_HISTORY: 'celestial_chat_history',
  ROBOT_CONFIG: 'celestial_robot_config',
  UI_STATE: 'celestial_ui_state',
  SHOW_LANDING: 'celestial_show_landing'
};

class StorageService {
  private db: IDBDatabase | null = null;

  /**
   * Initialize IndexedDB
   */
  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error('Failed to open IndexedDB:', request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        console.log('✅ IndexedDB initialized');
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Create object stores
        if (!db.objectStoreNames.contains(STORES.ALGORITHMS)) {
          db.createObjectStore(STORES.ALGORITHMS, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(STORES.SCENARIOS)) {
          db.createObjectStore(STORES.SCENARIOS, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(STORES.PARAMETERS)) {
          db.createObjectStore(STORES.PARAMETERS, { keyPath: 'algorithmId' });
        }

        console.log('✅ IndexedDB object stores created');
      };
    });
  }

  // ==================== localStorage Methods ====================

  /**
   * Save to localStorage with JSON serialization
   */
  saveLocal<T>(key: string, data: T): void {
    try {
      localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
      console.error(`Failed to save to localStorage (${key}):`, error);
    }
  }

  /**
   * Load from localStorage with JSON parsing
   */
  loadLocal<T>(key: string): T | null {
    try {
      const data = localStorage.getItem(key);
      return data ? JSON.parse(data) : null;
    } catch (error) {
      console.error(`Failed to load from localStorage (${key}):`, error);
      return null;
    }
  }

  /**
   * Remove from localStorage
   */
  removeLocal(key: string): void {
    localStorage.removeItem(key);
  }

  // ==================== IndexedDB Methods ====================

  /**
   * Save to IndexedDB
   */
  async saveIndexedDB(storeName: string, data: any): Promise<void> {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('IndexedDB not initialized'));
        return;
      }

      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(data);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Load from IndexedDB by key
   */
  async loadIndexedDB<T>(storeName: string, key: string): Promise<T | null> {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('IndexedDB not initialized'));
        return;
      }

      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(key);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Load all items from IndexedDB store
   */
  async loadAllIndexedDB<T>(storeName: string): Promise<T[]> {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('IndexedDB not initialized'));
        return;
      }

      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Delete from IndexedDB
   */
  async deleteIndexedDB(storeName: string, key: string): Promise<void> {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('IndexedDB not initialized'));
        return;
      }

      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  /**
   * Clear all data from IndexedDB store
   */
  async clearIndexedDB(storeName: string): Promise<void> {
    if (!this.db) {
      await this.init();
    }

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('IndexedDB not initialized'));
        return;
      }

      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.clear();

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // ==================== Utility Methods ====================

  /**
   * Clear all stored data (localStorage + IndexedDB)
   */
  async clearAll(): Promise<void> {
    // Clear localStorage
    Object.values(LOCAL_KEYS).forEach(key => {
      this.removeLocal(key);
    });

    // Clear IndexedDB
    await Promise.all([
      this.clearIndexedDB(STORES.ALGORITHMS),
      this.clearIndexedDB(STORES.SCENARIOS),
      this.clearIndexedDB(STORES.PARAMETERS)
    ]);

    console.log('✅ All storage cleared');
  }

  /**
   * Get storage usage estimate
   */
  async getStorageEstimate(): Promise<{
    usage: number;
    quota: number;
    percentage: number;
  }> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      return {
        usage: estimate.usage || 0,
        quota: estimate.quota || 0,
        percentage: ((estimate.usage || 0) / (estimate.quota || 1)) * 100
      };
    }
    return { usage: 0, quota: 0, percentage: 0 };
  }
}

// Export singleton instance
export const storageService = new StorageService();

// Export constants for use in components
export { LOCAL_KEYS, STORES };

// Initialize on import
storageService.init().catch(console.error);
