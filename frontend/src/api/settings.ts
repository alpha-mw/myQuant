import { api } from './client';
import type { SettingsResponse, SettingsUpdateRequest } from '../types/settings';

export function fetchSettings() {
  return api.get<SettingsResponse>('/api/settings/');
}

export function updateSettings(updates: SettingsUpdateRequest) {
  return api.patch<{ ok: boolean; updated: string[] }>('/api/settings/', updates);
}
