import AsyncStorage from '@react-native-async-storage/async-storage';

const HISTORY_KEY = '@voicespoof_history';
const MAX_HISTORY = 50;

export const HistoryStorage = {
  /** Save a new result. Returns the full updated history array. */
  async save(entry) {
    try {
      const existing = await HistoryStorage.load();
      const newEntry = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        ...entry,
      };
      const updated = [newEntry, ...existing].slice(0, MAX_HISTORY);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
      return updated;
    } catch (e) {
      console.error('[History] Save error:', e);
      return [];
    }
  },

  /** Load and parse history from storage. */
  async load() {
    try {
      const raw = await AsyncStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  },

  /** Delete a specific entry by id. */
  async delete(id) {
    try {
      const existing = await HistoryStorage.load();
      const updated = existing.filter(e => e.id !== id);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
      return updated;
    } catch (e) {
      return [];
    }
  },

  /** Clear all history. */
  async clear() {
    try {
      await AsyncStorage.removeItem(HISTORY_KEY);
    } catch (e) {}
  },
};

export const SettingsStorage = {
  BACKEND_KEY: '@voicespoof_backend_url',

  async getUrl(fallback) {
    try {
      const val = await AsyncStorage.getItem(SettingsStorage.BACKEND_KEY);
      return val || fallback;
    } catch {
      return fallback;
    }
  },

  async setUrl(url) {
    try {
      await AsyncStorage.setItem(SettingsStorage.BACKEND_KEY, url);
    } catch {}
  },
};
