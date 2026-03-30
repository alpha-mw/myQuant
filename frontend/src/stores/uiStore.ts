import { create } from 'zustand'

interface UIState {
  configPanelCollapsed: boolean
  activeJobId: string | null
  setConfigPanelCollapsed: (v: boolean) => void
  setActiveJobId: (id: string | null) => void
}

export const useUIStore = create<UIState>((set) => ({
  configPanelCollapsed: false,
  activeJobId: null,
  setConfigPanelCollapsed: (v) => set({ configPanelCollapsed: v }),
  setActiveJobId: (id) => set({ activeJobId: id }),
}))
