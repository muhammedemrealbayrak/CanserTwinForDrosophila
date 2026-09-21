// ============================================================================
// DROSOPHILA IN SILICO DIGITAL TWIN - UNITY 3D / C# .NET ENGINE
// File: DrosophilaConnectomeSimulator.cs
// Namespace: DrosophilaTwin.Core
// Author: Muhammed Emre Albayrak
// License: MIT
// ============================================================================

using System;
using System.Collections.Generic;

namespace DrosophilaTwin.Core
{
    public class DrosophilaConnectomeSimulator
    {
        public List<NeuronAgent> Neurons { get; private set; } = new List<NeuronAgent>();
        public float GlobalCholinergicTone { get; private set; } = 1.0f;
        public float AverageCalciumNm { get; private set; } = 100.0f;
        public int LastStepSpikeCount { get; private set; } = 0;

        private Random _rng = new Random(42);

        // Izhikevich parameters (Regular Spiking / Excitatory & Fast Spiking / Inhibitory)
        private const float ParamA = 0.02f;
        private const float ParamB = 0.2f;
        private const float ParamC = -65.0f;
        private const float ParamD = 8.0f;
        private const float SpikeThreshold = 30.0f;

        public DrosophilaConnectomeSimulator(int neuronCount = 500)
        {
            InitializeBrainNetwork(neuronCount);
        }

        public void InitializeBrainNetwork(int count)
        {
            Neurons.Clear();
            var regions = (NeuropilRegion[])Enum.GetValues(typeof(NeuropilRegion));

            for (int i = 0; i < count; i++)
            {
                NeuropilRegion region = regions[i % regions.Length];
                Vector3D pos = GenerateAnatomicalCoordinates(region);

                // 70% Acetylcholine (excitatory in insects), 20% GABA, 10% Modulatory
                NeurotransmitterType nt;
                double r = _rng.NextDouble();
                if (r < 0.70) nt = NeurotransmitterType.Acetylcholine;
                else if (r < 0.90) nt = NeurotransmitterType.GABA;
                else nt = NeurotransmitterType.Octopamine;

                var neuron = new NeuronAgent
                {
                    id = i,
                    label = $"Neu_{region}_{i}",
                    region = region,
                    transmitter = nt,
                    position = pos,
                    membranePotentialMv = -65.0f + (float)(_rng.NextDouble() * 5.0 - 2.5),
                    recoveryVariable = -65.0f * ParamB,
                    intracellularCalciumNm = 90.0f + (float)(_rng.NextDouble() * 20.0),
                    isSpiking = false,
                    synapticWeightSum = 0.0f
                };

                Neurons.Add(neuron);
            }

            // Synaptic wiring (Connectome Graph)
            for (int i = 0; i < count; i++)
            {
                int synapseCount = _rng.Next(3, 10);
                for (int s = 0; s < synapseCount; s++)
                {
                    int target = _rng.Next(0, count);
                    if (target != i && !Neurons[i].downstreamSynapseIds.Contains(target))
                    {
                        Neurons[i].downstreamSynapseIds.Add(target);
                    }
                }
            }
        }

        public void Step(float dtMs, float externalStimulusCurrent = 0.0f)
        {
            int spikes = 0;
            float totalCa = 0.0f;
            int achSpikes = 0;

            // 1. Synaptic Integration and Membrane Dynamics
            for (int i = 0; i < Neurons.Count; i++)
            {
                var n = Neurons[i];
                float iSyn = n.synapticWeightSum + externalStimulusCurrent;
                
                // Stochastic spontaneous sensory background input
                if (_rng.NextDouble() < 0.05)
                {
                    iSyn += (float)(_rng.NextDouble() * 12.0 + 3.0);
                }

                // Izhikevich 2-variable ODE Integration (Euler dt)
                float dtSec = dtMs / 1000.0f;
                float v = n.membranePotentialMv;
                float u = n.recoveryVariable;

                // Two half-steps for numerical stability
                for (int sub = 0; sub < 2; sub++)
                {
                    float halfDt = (dtMs * 0.5f);
                    v += halfDt * (0.04f * v * v + 5.0f * v + 140.0f - u + iSyn);
                    u += halfDt * (ParamA * (ParamB * v - u));
                }

                if (v >= SpikeThreshold)
                {
                    n.isSpiking = true;
                    n.membranePotentialMv = ParamC;
                    n.recoveryVariable = u + ParamD;
                    spikes++;

                    // Calcium transient surge upon action potential
                    n.intracellularCalciumNm = Math.Min(1200.0f, n.intracellularCalciumNm + 180.0f);

                    if (n.transmitter == NeurotransmitterType.Acetylcholine)
                    {
                        achSpikes++;
                    }

                    // Propagate spike to downstream targets
                    float synapticWeight = (n.transmitter == NeurotransmitterType.GABA) ? -4.5f : 3.8f;
                    foreach (int targetId in n.downstreamSynapseIds)
                    {
                        if (targetId >= 0 && targetId < Neurons.Count)
                        {
                            Neurons[targetId].synapticWeightSum += synapticWeight;
                        }
                    }
                }
                else
                {
                    n.isSpiking = false;
                    n.membranePotentialMv = v;
                    n.recoveryVariable = u;
                    
                    // Calcium decay back to basal (80-100 nM)
                    n.intracellularCalciumNm = Math.Max(85.0f, n.intracellularCalciumNm - (n.intracellularCalciumNm - 85.0f) * 0.08f);
                }

                // Decay synaptic inputs
                n.synapticWeightSum *= 0.70f;
                totalCa += n.intracellularCalciumNm;
            }

            LastStepSpikeCount = spikes;
            AverageCalciumNm = totalCa / Math.Max(1, Neurons.Count);

            // Calculate systemic cholinergic reflex tone
            float rawTone = (float)achSpikes / Math.Max(1, spikes);
            GlobalCholinergicTone = 0.85f * GlobalCholinergicTone + 0.15f * (rawTone * 2.5f);
            GlobalCholinergicTone = Math.Max(0.2f, Math.Min(2.5f, GlobalCholinergicTone));
        }

        private Vector3D GenerateAnatomicalCoordinates(NeuropilRegion region)
        {
            // Anatomical 3D bounds in Drosophila brain coordinates (microns)
            switch (region)
            {
                case NeuropilRegion.AntennalLobe_Left:
                    return new Vector3D(-70f + RandF(-15, 15), 110f + RandF(-20, 20), -30f + RandF(-15, 15));
                case NeuropilRegion.AntennalLobe_Right:
                    return new Vector3D(70f + RandF(-15, 15), 110f + RandF(-20, 20), -30f + RandF(-15, 15));
                case NeuropilRegion.MushroomBody_Left:
                    return new Vector3D(-120f + RandF(-25, 25), 40f + RandF(-30, 30), 20f + RandF(-20, 20));
                case NeuropilRegion.MushroomBody_Right:
                    return new Vector3D(120f + RandF(-25, 25), 40f + RandF(-30, 30), 20f + RandF(-20, 20));
                case NeuropilRegion.CentralComplex_FanShapedBody:
                    return new Vector3D(RandF(-25, 25), 15f + RandF(-15, 15), 40f + RandF(-15, 15));
                case NeuropilRegion.CentralComplex_EllipsoidBody:
                    return new Vector3D(RandF(-18, 18), 35f + RandF(-12, 12), 20f + RandF(-12, 12));
                case NeuropilRegion.OpticLobe_Left:
                    return new Vector3D(-210f + RandF(-35, 35), -10f + RandF(-40, 40), RandF(-30, 30));
                case NeuropilRegion.OpticLobe_Right:
                    return new Vector3D(210f + RandF(-35, 35), -10f + RandF(-40, 40), RandF(-30, 30));
                case NeuropilRegion.SubesophagealZone:
                    return new Vector3D(RandF(-40, 40), -90f + RandF(-25, 25), -60f + RandF(-20, 20));
                case NeuropilRegion.ParsIntercerebralis_Neuroendocrine:
                    return new Vector3D(RandF(-20, 20), 80f + RandF(-15, 15), 50f + RandF(-15, 15));
                default:
                    return new Vector3D(RandF(-50, 50), RandF(-50, 50), RandF(-50, 50));
            }
        }

        private float RandF(float min, float max)
        {
            return (float)(min + _rng.NextDouble() * (max - min));
        }
    }
}
