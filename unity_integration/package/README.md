# Drosophila In Silico Digital Twin - Unity 3D / C# .NET Engine

[![Unity](https://img.shields.io/badge/Unity-2021.3%2B-blue.svg)](https://unity.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Connectome](https://img.shields.io/badge/FlyWire-139k%20Neurons-cyan.svg)](https://flywire.ai/)

Unity 3D / C# .NET integration for the Drosophila Cancer Digital Twin Platform. Simulates whole-brain neural connectome circuits, neuro-immune reflex communication, spatial 3D tumor growth, macrophage hemocyte infiltration, and pharmacokinetics in real-time.

---

## 📂 Architecture Overview

```
unity_integration/
├── csharp/
│   ├── DigitalTwinTypes.cs                # Strongly typed telemetry, agents, and data contracts
│   ├── DrosophilaConnectomeSimulator.cs   # Izhikevich spiking neural dynamics & FlyWire neuropils
│   ├── TumorAgentSystem.cs                # 3D spatial agent-based tumor & macrophage hemocyte immune system
│   ├── DrugPharmacokinetics.cs           # 2-compartment PK/PD, Hill receptor occupancy & kinase inhibition
│   └── SimulationBridge.cs               # Unity MonoBehaviour controller with interactive inputs
└── package/
    ├── package.json                       # Unity Package Manager (UPM) manifest
    └── README.md                          # Quickstart guide
```

---

## 🚀 Quickstart Guide for Unity 3D

### Method 1: Unity Package Manager (Git URL)
1. In Unity Editor, go to **Window** ➔ **Package Manager**.
2. Click **`+`** ➔ **Add package from git URL...**
3. Enter:
   ```
   https://github.com/muhammedemrealbayrak/CanserTwinForDrosophila.git?path=/unity_integration/package
   ```

### Method 2: Direct Script Import
Simply copy the `unity_integration/csharp/` folder into your Unity project's `Assets/Scripts/DrosophilaTwin/` directory.

---

## 🎮 Interactive Scene Controls
Attach `SimulationBridge.cs` to an empty GameObject in your scene:
- **`Space`**: Administer $5\text{ }\mu\text{M}$ bolus dose of active compound.
- **`1`**: Switch to **`DeNovo_Champion`** (High affinity nAChR & MEK soft-drug).
- **`2`**: Switch to **`Cisplatin`** (Traditional cytotoxic DNA cross-linker).
- **`3`**: Switch to **`Trametinib`** (Sub-nanomolar MEK1 inhibitor).
- **`4`**: Switch to **`Multimodal_Synergy_Cocktail`** (Synergistic multi-target rescue).
- **`R`**: Reset digital twin state.

---

## 🔬 Mathematical Specifications
- **Neural Dynamics**: Izhikevich 2-variable non-linear ODE:
  $$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I_{\text{syn}}$$
  $$\frac{du}{dt} = a(bv - u)$$
- **Receptor Binding**: Hill Equation:
  $$\theta = \frac{C_{\text{brain}}^h}{C_{\text{brain}}^h + K_d^h}$$
- **Immunological Clearance**: Macrophage phagocytosis modulated by CD47 'Don't-Eat-Me' disruption:
  $$P_{\text{engulf}} = 1.0 - \left(\text{CD47}_{\text{expr}} \cdot (1 - \text{BreachFactor})\right)$$
