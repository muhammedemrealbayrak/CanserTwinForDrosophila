"""
Drosophila In Silico Neuro-Immune Digital Twin
Modül: pipeline/flywire_circuit.py
======================================================
Yazar: Nöro-Bilişim & Konektom Dinamikleri Ekibi
Açıklama:
    FlyWire konektom devresi ve milisaniyelik sinyal iletim motoru.
    Duyu nöronu (~10 ms), ara nöron (~25 ms) ve beyin bağışıklık merkezi
    (~150 ms) üzerinden toplam 185 ms gecikmeli eferent iletimi çözer.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
import numpy as np


@dataclass
class CircuitState:
    """Nöral devrenin ve FlyWire KCg-m nöronunun anlık milisaniyelik durumu."""
    sensory_voltage_mv: float = -65.0      # İstirahat potansiyeli (-65 mV)
    interneuron_voltage_mv: float = -65.0
    kcg_membrane_mv: float = -65.0         # FlyWire KCg-m Kenyon Hücresi Zar Potansiyeli (mV)
    kcg_calcium_nm: float = 100.0          # Hücre içi Kalsiyum [Ca2+]i konsantrasyonu (nM)
    snpf_release_rate: float = 0.0         # sNPF nöropeptit salınım oranı (%/s)
    central_activation_index: float = 0.0  # Beyin merkezi aktivasyon derecesi [0.0 - 1.0]
    efferent_firing_hz: float = 0.0        # Lenf bezine giden sinirsel ateşleme frekansı (Hz)
    acetylcholine_quanta_nm: float = 5.0   # Salınan nörotransmitter konsantrasyonu (nM)
    spike_event: bool = False              # Anlık aksiyon potansiyeli tepe noktası vurdu mu?


class FlyWireCircuitSimulator:
    """
    Milisaniyelik nöral iletim devresi simülatörü.
    """

    def __init__(
        self,
        sensory_latency_ms: float = 10.0,
        interneuron_latency_ms: float = 25.0,
        central_latency_ms: float = 150.0,
        resonant_frequency_hz: float = 45.0
    ):
        self.sensory_latency_ms = sensory_latency_ms
        self.interneuron_latency_ms = interneuron_latency_ms
        self.central_latency_ms = central_latency_ms
        self.resonant_frequency_hz = resonant_frequency_hz
        self.total_loop_latency_ms = sensory_latency_ms + interneuron_latency_ms + central_latency_ms

        self.state = CircuitState()
        self.sensory_timer_ms: float = 0.0
        self.inter_timer_ms: float = 0.0
        self.central_timer_ms: float = 0.0

    def step_circuit_ms(self, ligand_binding_potency: float, dt_ms: float = 1.0) -> CircuitState:
        """
        Milisaniye çözünürlüğünde nöral aksiyon potansiyeli ve sinaptik iletim adımı.
        
        Args:
            ligand_binding_potency: İlacın duyu reseptörüne bağlanma şiddeti [0.0 - 1.0].
            dt_ms: Alt-adım süresi (varsayılan: 1.0 ms).
        """
        # 1. Duyu Nöronu Depolarizasyonu (~10 ms eşik)
        if ligand_binding_potency > 0.15:
            self.sensory_timer_ms += dt_ms
            if self.sensory_timer_ms >= self.sensory_latency_ms:
                self.state.sensory_voltage_mv = -65.0 + 95.0 * np.clip(ligand_binding_potency, 0.0, 1.0)
        else:
            self.sensory_timer_ms = max(0.0, self.sensory_timer_ms - dt_ms * 0.5)
            self.state.sensory_voltage_mv += (-65.0 - self.state.sensory_voltage_mv) * 0.05

        # 2. Ara Nöron Geçişi (~25 ms eşik)
        sensory_firing = (self.state.sensory_voltage_mv + 65.0) / 95.0
        if sensory_firing > 0.25:
            self.inter_timer_ms += dt_ms
            if self.inter_timer_ms >= self.interneuron_latency_ms:
                self.state.interneuron_voltage_mv = -65.0 + 90.0 * sensory_firing
        else:
            self.inter_timer_ms = max(0.0, self.inter_timer_ms - dt_ms * 0.5)
            self.state.interneuron_voltage_mv += (-65.0 - self.state.interneuron_voltage_mv) * 0.03

        # 3. FlyWire KCg-m (Kenyon Cell Gamma Lobe #720575940608530955) Elektrofizyolojisi
        # Calyx'teki kolinerjik/dopaminerjik girdilere yanıt olarak Izhikevich / Hodgkin-Huxley benzeri spiking
        input_current = max(0.0, (self.state.interneuron_voltage_mv + 60.0) * 0.8)
        
        # Zar potansiyeli dinamiği
        spike_threshold_mv = -42.0
        self.state.spike_event = False

        if self.state.kcg_membrane_mv >= spike_threshold_mv:
            # Aksiyon potansiyeli tepe vuruşu (+25 mV) ve ardından repolarizasyon
            self.state.kcg_membrane_mv = -72.0  # After-hyperpolarization (AHP)
            self.state.spike_event = True
            # Kalsiyum akısı (VGCC: Voltaj kapılı Ca2+ kanalları açılır)
            self.state.kcg_calcium_nm = min(1400.0, self.state.kcg_calcium_nm + 120.0)
        else:
            # Sızıntı ve sinaptik akım integrasyonu
            dv = ((-65.0 - self.state.kcg_membrane_mv) * 0.12 + input_current * 1.8) * (dt_ms / 5.0)
            self.state.kcg_membrane_mv = float(np.clip(self.state.kcg_membrane_mv + dv, -80.0, +25.0))
            if self.state.kcg_membrane_mv > spike_threshold_mv:
                self.state.kcg_membrane_mv = +24.5  # Spike peak

        # Kalsiyum temizleme pompası (PMCA / SERCA: tau = 300 ms)
        ca_decay = (100.0 - self.state.kcg_calcium_nm) * 0.02 * (dt_ms / 5.0)
        self.state.kcg_calcium_nm = max(100.0, self.state.kcg_calcium_nm + ca_decay)

        # sNPF (short Neuropeptide F) ve Asetilkolin (ACh) Vezikül Salınımı
        # Ca2+ > 350 nM olduğunda dens-core vezikül ekzositozu tetiklenir
        if self.state.kcg_calcium_nm > 350.0:
            ca_norm = (self.state.kcg_calcium_nm - 350.0) / 750.0
            self.state.snpf_release_rate = float(np.clip(ca_norm * 85.0, 0.0, 100.0))
        else:
            self.state.snpf_release_rate *= 0.90

        # 4. Beyin Bağışıklık Merkezi / SEZ & Ventral Kordon Eferent İletimi (~150 ms gecikme)
        kcg_drive = self.state.snpf_release_rate / 100.0
        if kcg_drive > 0.15 or sensory_firing > 0.4:
            self.central_timer_ms += dt_ms
            if self.central_timer_ms >= self.central_latency_ms:
                self.state.central_activation_index = float(np.clip(self.state.central_activation_index + 0.09, 0.0, 1.0))
        else:
            self.central_timer_ms = max(0.0, self.central_timer_ms - dt_ms * 0.2)
            self.state.central_activation_index = max(0.0, self.state.central_activation_index - 0.006)

        # 5. Eferent Vagus / Ventral Kordon Ateşleme Frekansı (Hz)
        if self.state.central_activation_index > 0.1:
            target_freq = self.resonant_frequency_hz * self.state.central_activation_index
            self.state.efferent_firing_hz += (target_freq - self.state.efferent_firing_hz) * 0.12
        else:
            self.state.efferent_firing_hz *= 0.92

        # Asetilkolin (ACh) kuantumu (nM)
        freq_ratio = self.state.efferent_firing_hz / self.resonant_frequency_hz
        self.state.acetylcholine_quanta_nm = float(5.0 + 95.0 * np.clip(freq_ratio, 0.0, 1.8))

        return self.state

    def simulate_macro_second_response(self, ligand_potency: float, duration_s: float = 1.0) -> Dict[str, Any]:
        """
        1 saniyelik makro zaman dilimi boyunca alt-adımları çözer ve elektrofizyoloji izlerini üretir.
        """
        dt_ms = 4.0
        steps = int((duration_s * 1000.0) / dt_ms)
        voltage_trace = []
        calcium_trace = []
        spike_count = 0

        for _ in range(steps):
            st = self.step_circuit_ms(ligand_potency, dt_ms=dt_ms)
            if len(voltage_trace) < 50:
                voltage_trace.append(round(st.kcg_membrane_mv, 1))
                calcium_trace.append(round(st.kcg_calcium_nm, 1))
            if st.spike_event:
                spike_count += 1

        # Kemik iliği / lenf bezi uyarım çarpanı (Hill doygunluk denklemi)
        ka = 32.0  # nM ACh eşik sabiti
        drive = 2.5 * (self.state.acetylcholine_quanta_nm / (self.state.acetylcholine_quanta_nm + ka))
        
        return {
            "efferent_firing_hz": round(self.state.efferent_firing_hz, 1),
            "marrow_stimulation_drive": round(float(drive), 3),
            "kcg_membrane_mv": round(self.state.kcg_membrane_mv, 1),
            "kcg_calcium_nm": round(self.state.kcg_calcium_nm, 1),
            "snpf_release_pct": round(self.state.snpf_release_rate, 1),
            "ach_quanta_nm": round(self.state.acetylcholine_quanta_nm, 1),
            "spikes_per_sec": spike_count,
            "voltage_trace": voltage_trace,
            "calcium_trace": calcium_trace
        }
