// ============================================================================
// DROSOPHILA IN SILICO DIGITAL TWIN - UNITY 3D / C# .NET ENGINE
// File: SimulationBridge.cs
// Namespace: DrosophilaTwin.Unity
// Author: Muhammed Emre Albayrak
// License: MIT
// ============================================================================

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using DrosophilaTwin.Core;

namespace DrosophilaTwin.Unity
{
    public class SimulationBridge : MonoBehaviour
    {
        [Header("Digital Twin Core Configuration")]
        [Tooltip("Number of simulated FlyWire connectome neurons")]
        public int initialNeuronCount = 400;

        [Tooltip("Initial number of transformed cancer cells")]
        public int initialTumorCells = 50;

        [Tooltip("Initial patrolling macrophage hemocytes")]
        public int initialHemocytes = 25;

        [Tooltip("Simulation time-step in milliseconds per frame")]
        public float timeStepMs = 1.0f;

        [Header("Active Therapeutic Agent")]
        public string activeCompound = "DeNovo_Champion";
        public float initialDoseUm = 10.0f;

        [Header("3D Visual Representation Prefabs")]
        public GameObject neuronPrefab;
        public GameObject tumorCellPrefab;
        public GameObject hemocytePrefab;

        [Header("Telemetry & Live Output")]
        [TextArea(3, 10)]
        public string latestJsonTelemetry;

        // Core Simulator Subsystems
        public DrosophilaConnectomeSimulator Connectome { get; private set; }
        public TumorAgentSystem TumorSystem { get; private set; }
        public DrugPharmacokinetics Pharmacokinetics { get; private set; }

        private long _stepCounter = 0;
        private float _simTimeMinutes = 0.0f;

        // Visual GameObject tracking in Unity Scene
        private Dictionary<int, Transform> _neuronVisuals = new Dictionary<int, Transform>();
        private Dictionary<int, Transform> _tumorVisuals = new Dictionary<int, Transform>();
        private Dictionary<int, Transform> _hemocyteVisuals = new Dictionary<int, Transform>();

        private void Awake()
        {
            InitializeDigitalTwin();
        }

        public void InitializeDigitalTwin()
        {
            Debug.Log("<color=#00f0ff>[DrosophilaTwin]</color> Initializing In Silico Digital Twin Engine...");

            // 1. Initialize Neural Connectome
            Connectome = new DrosophilaConnectomeSimulator(initialNeuronCount);

            // 2. Initialize Tumor Microenvironment at Mushroom Body / Glial Niche
            Vector3D primaryTumorNiche = new Vector3D(-120f, 40f, 20f);
            TumorSystem = new TumorAgentSystem(primaryTumorNiche, initialTumorCells, initialHemocytes);

            // 3. Initialize Pharmacokinetics Engine
            Pharmacokinetics = new DrugPharmacokinetics(activeCompound);
            if (initialDoseUm > 0.0f)
            {
                Pharmacokinetics.AdministerBolusDose(initialDoseUm);
            }

            // 4. Instantiate Visual Actors in 3D Scene (if prefabs assigned)
            InstantiateVisualHierarchy();

            Debug.Log($"<color=#00ff9d>[DrosophilaTwin]</color> Initialized with {Connectome.Neurons.Count} Neurons, {TumorSystem.TumorCells.Count} Tumor Cells.");
        }

        private void Update()
        {
            // Handle Interactive Keyboard Inputs
            if (Input.GetKeyDown(KeyCode.Space))
            {
                AdministerDrugDose(5.0f);
            }
            if (Input.GetKeyDown(KeyCode.Alpha1)) SwitchCompound("DeNovo_Champion");
            if (Input.GetKeyDown(KeyCode.Alpha2)) SwitchCompound("Cisplatin");
            if (Input.GetKeyDown(KeyCode.Alpha3)) SwitchCompound("Trametinib");
            if (Input.GetKeyDown(KeyCode.Alpha4)) SwitchCompound("Multimodal_Synergy_Cocktail");
            if (Input.GetKeyDown(KeyCode.R)) InitializeDigitalTwin();

            // Run Physics / Biological Integration Step
            StepSimulation();

            // Update Scene Visuals
            UpdateVisualTransforms();
        }

        public void StepSimulation()
        {
            _stepCounter++;
            float dtHours = (timeStepMs * 0.05f) / 60.0f; // Scaled biological time
            _simTimeMinutes += (timeStepMs * 0.05f);

            // A. Update Pharmacokinetics
            Pharmacokinetics.Step(dtHours);

            // B. Update Connectome (Izhikevich ODEs)
            Connectome.Step(timeStepMs);

            // C. Update Tumor and Hemocyte Microenvironment
            TumorSystem.Step(
                dtHours,
                Pharmacokinetics.CytotoxicRatePerHour,
                Pharmacokinetics.Cd47ShieldBreachFactor,
                Connectome.GlobalCholinergicTone
            );

            // D. Emit JSON Telemetry Frame
            latestJsonTelemetry = GenerateTelemetryJson();
        }

        public void AdministerDrugDose(float doseUm)
        {
            Pharmacokinetics.AdministerBolusDose(doseUm);
            Debug.Log($"<color=#ffb703>[DrosophilaTwin]</color> Administered {doseUm} µM bolus of {Pharmacokinetics.ActiveCompoundName}.");
        }

        public void SwitchCompound(string compoundName)
        {
            Pharmacokinetics.SetCompound(compoundName);
            Debug.Log($"<color=#00f0ff>[DrosophilaTwin]</color> Switched active agent to: {compoundName}");
        }

        public string GenerateTelemetryJson()
        {
            var frame = new DigitalTwinTelemetryFrame
            {
                step = _stepCounter,
                simulationTimeMinutes = _simTimeMinutes,
                viableNeuronCount = Connectome.Neurons.Count,
                activeSpikesInStep = Connectome.LastStepSpikeCount,
                totalTumorCellCount = TumorSystem.TumorCells.Count,
                proliferatingTumorCount = TumorSystem.TumorCells.FindAll(c => c.state == TumorCellState.Proliferating).Count,
                apoptoticTumorCount = TumorSystem.CumulativeApoptoticCount,
                activeHemocyteCount = TumorSystem.Hemocytes.FindAll(h => h.state != HemocyteState.ExhaustedByTumorShield).Count,
                averageBrainCalciumNm = Connectome.AverageCalciumNm,
                microenvironmentLactateMm = TumorSystem.SystemicLactateMm,
                cachexiaScorePercent = TumorSystem.CachexiaScore,
                pkState = Pharmacokinetics.GetTelemetry()
            };

            return JsonUtility.ToJson(frame, true);
        }

        private void InstantiateVisualHierarchy()
        {
            // Clear existing visuals
            foreach (var kvp in _neuronVisuals) if (kvp.Value) Destroy(kvp.Value.gameObject);
            foreach (var kvp in _tumorVisuals) if (kvp.Value) Destroy(kvp.Value.gameObject);
            foreach (var kvp in _hemocyteVisuals) if (kvp.Value) Destroy(kvp.Value.gameObject);
            _neuronVisuals.Clear();
            _tumorVisuals.Clear();
            _hemocyteVisuals.Clear();

            // Instantiate Neurons
            if (neuronPrefab != null)
            {
                Transform parent = new GameObject("NeuronAgents").transform;
                parent.SetParent(this.transform);
                foreach (var n in Connectome.Neurons)
                {
                    GameObject obj = Instantiate(neuronPrefab, new Vector3(n.position.x, n.position.y, n.position.z), Quaternion.identity, parent);
                    obj.name = n.label;
                    _neuronVisuals[n.id] = obj.transform;
                }
            }

            // Instantiate Tumor Cells
            if (tumorCellPrefab != null)
            {
                Transform parent = new GameObject("TumorAgents").transform;
                parent.SetParent(this.transform);
                foreach (var t in TumorSystem.TumorCells)
                {
                    GameObject obj = Instantiate(tumorCellPrefab, new Vector3(t.position.x, t.position.y, t.position.z), Quaternion.identity, parent);
                    obj.name = $"Tumor_{t.id}";
                    _tumorVisuals[t.id] = obj.transform;
                }
            }

            // Instantiate Hemocytes
            if (hemocytePrefab != null)
            {
                Transform parent = new GameObject("HemocyteAgents").transform;
                parent.SetParent(this.transform);
                foreach (var h in TumorSystem.Hemocytes)
                {
                    GameObject obj = Instantiate(hemocytePrefab, new Vector3(h.position.x, h.position.y, h.position.z), Quaternion.identity, parent);
                    obj.name = $"Hemocyte_{h.id}";
                    _hemocyteVisuals[h.id] = obj.transform;
                }
            }
        }

        private void UpdateVisualTransforms()
        {
            // Dynamically synchronize hemocytes position and color
            foreach (var h in TumorSystem.Hemocytes)
            {
                if (_hemocyteVisuals.TryGetValue(h.id, out Transform tr) && tr != null)
                {
                    tr.position = Vector3.Lerp(tr.position, new Vector3(h.position.x, h.position.y, h.position.z), Time.deltaTime * 8.0f);
                }
            }

            // Update tumor cells (scale down / destroy on apoptosis)
            foreach (var t in TumorSystem.TumorCells)
            {
                if (_tumorVisuals.TryGetValue(t.id, out Transform tr) && tr != null)
                {
                    if (t.state == TumorCellState.Apoptotic || t.state == TumorCellState.EngulfedByHemocyte)
                    {
                        tr.localScale = Vector3.Lerp(tr.localScale, Vector3.zero, Time.deltaTime * 4.0f);
                    }
                }
            }
        }
    }
}
