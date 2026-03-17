import { api } from './client';
import type { SettingsResponse } from '../types/api';

export function fetchSettings() {
  return api.get<SettingsResponse>('/settings');
}

export function updateSettings(updates: Record<string, unknown>) {
  return api.put<SettingsResponse>('/settings', updates);
}
