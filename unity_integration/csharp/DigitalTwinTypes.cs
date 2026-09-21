// ============================================================================
// DROSOPHILA IN SILICO DIGITAL TWIN - UNITY 3D / C# .NET ENGINE
// File: DigitalTwinTypes.cs
// Namespace: DrosophilaTwin.Core
// Author: Muhammed Emre Albayrak
// License: MIT
// ============================================================================

using System;
using System.Collections.Generic;

namespace DrosophilaTwin.Core
{
    [Serializable]
    public enum NeuropilRegion
    {
        AntennalLobe_Left,
        AntennalLobe_Right,
        MushroomBody_Left,
        MushroomBody_Right,
        CentralComplex_FanShapedBody,
        CentralComplex_EllipsoidBody,
        OpticLobe_Left,
        OpticLobe_Right,
        SubesophagealZone,
        ParsIntercerebralis_Neuroendocrine
    }

    [Serializable]
    public enum NeurotransmitterType
    {
        Acetylcholine,  // Excitatory / Nicotinic
        GABA,           // Inhibitory
        Glutamate,      // Dual / Motoneuron
        Dopamine,       // Modulatory / Mushroom Body
        Octopamine      // Adrenergic-like / Stress & Immune
    }

    [Serializable]
    public enum TumorCellState
    {
        CancerStemCell,
        Proliferating,
        Hypoxic,
        Apoptotic,
        EngulfedByHemocyte,
        Necrotic
    }

    [Serializable]
    public enum HemocyteState
    {
        Patrolling,
        PrimedByCytokines,
        ActivelyEngulfing,
        ExhaustedByTumorShield
    }

    [Serializable]
    public struct Vector3D
    {
        public float x;
        public float y;
        public float z;

        public Vector3D(float x, float y, float z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public float DistanceTo(Vector3D other)
        {
            float dx = this.x - other.x;
            float dy = this.y - other.y;
            float dz = this.z - other.z;
            return (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    [Serializable]
    public class NeuronAgent
    {
        public int id;
        public string label;
        public NeuropilRegion region;
        public NeurotransmitterType transmitter;
        public Vector3D position;
        public float membranePotentialMv;  // -70.0 to +30.0 mV
        public float recoveryVariable;     // Izhikevich u
        public float intracellularCalciumNm; // 80.0 to 1200.0 nM
        public bool isSpiking;
        public float synapticWeightSum;
        public List<int> downstreamSynapseIds = new List<int>();
    }

    [Serializable]
    public class TumorCellAgent
    {
        public int id;
        public Vector3D position;
        public TumorCellState state;
        public float vitality;            // 0.0 to 1.0
        public float chemoResistance;      // 0.0 to 1.0 (e.g. Cisplatin resistance)
        public float cd47Expression;       // 'Don't-Eat-Me' immune evasion shield
        public float lactateProduction;    // Warburg acidity factor
        public float timeInHypoxia;
    }

    [Serializable]
    public class HemocyteAgent
    {
        public int id;
        public Vector3D position;
        public HemocyteState state;
        public float phagocyticCapacity;   // Engulfment reserve remaining
        public int? targetTumorCellId;
        public float speed;
    }

    [Serializable]
    public class PharmacokineticTelemetry
    {
        public string compoundName;
        public float plasmaConcentrationUm;
        public float brainIntercellularConcentrationUm;
        public float receptorOccupancyRatio; // 0.0 to 1.0 (Hill Equation)
        public float targetKinaseInhibitionPct;
        public float cytotoxicRatePerHour;
    }

    [Serializable]
    public class DigitalTwinTelemetryFrame
    {
        public long step;
        public float simulationTimeMinutes;
        public int viableNeuronCount;
        public int activeSpikesInStep;
        public int totalTumorCellCount;
        public int proliferatingTumorCount;
        public int apoptoticTumorCount;
        public int activeHemocyteCount;
        public float averageBrainCalciumNm;
        public float microenvironmentLactateMm;
        public float cachexiaScorePercent;
        public PharmacokineticTelemetry pkState;
    }
}
