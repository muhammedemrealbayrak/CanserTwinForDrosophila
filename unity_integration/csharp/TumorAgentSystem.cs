// ============================================================================
// DROSOPHILA IN SILICO DIGITAL TWIN - UNITY 3D / C# .NET ENGINE
// File: TumorAgentSystem.cs
// Namespace: DrosophilaTwin.Core
// Author: Muhammed Emre Albayrak
// License: MIT
// ============================================================================

using System;
using System.Collections.Generic;

namespace DrosophilaTwin.Core
{
    public class TumorAgentSystem
    {
        public List<TumorCellAgent> TumorCells { get; private set; } = new List<TumorCellAgent>();
        public List<HemocyteAgent> Hemocytes { get; private set; } = new List<HemocyteAgent>();

        public float SystemicLactateMm { get; private set; } = 1.2f;
        public float CachexiaScore { get; private set; } = 5.0f; // 0 to 100%
        public int CumulativeApoptoticCount { get; private set; } = 0;

        private Vector3D _tumorCentroid;
        private Random _rng = new Random(1337);
        private int _nextTumorId = 0;
        private int _nextHemocyteId = 0;

        public TumorAgentSystem(Vector3D primarySite, int initialTumorCount = 60, int initialHemocyteCount = 25)
        {
            _tumorCentroid = primarySite;
            SeedTumor(initialTumorCount);
            SeedHemocytes(initialHemocyteCount);
        }

        public void SeedTumor(int count)
        {
            TumorCells.Clear();
            _nextTumorId = 0;

            for (int i = 0; i < count; i++)
            {
                // Spherical normal distribution around primary oncogenic transformation site
                Vector3D offset = SampleSphereGaussian(25.0f);
                Vector3D pos = new Vector3D(
                    _tumorCentroid.x + offset.x,
                    _tumorCentroid.y + offset.y,
                    _tumorCentroid.z + offset.z
                );

                bool isStem = (i < 5);
                var cell = new TumorCellAgent
                {
                    id = _nextTumorId++,
                    position = pos,
                    state = isStem ? TumorCellState.CancerStemCell : TumorCellState.Proliferating,
                    vitality = 1.0f,
                    chemoResistance = isStem ? 0.75f : 0.20f,
                    cd47Expression = 0.85f, // High immune evasion by default
                    lactateProduction = 0.08f,
                    timeInHypoxia = 0.0f
                };

                TumorCells.Add(cell);
            }
        }

        public void SeedHemocytes(int count)
        {
            Hemocytes.Clear();
            _nextHemocyteId = 0;

            for (int i = 0; i < count; i++)
            {
                Vector3D offset = SampleSphereGaussian(80.0f);
                var h = new HemocyteAgent
                {
                    id = _nextHemocyteId++,
                    position = new Vector3D(_tumorCentroid.x + offset.x, _tumorCentroid.y + offset.y, _tumorCentroid.z + offset.z),
                    state = HemocyteState.Patrolling,
                    phagocyticCapacity = 5.0f,
                    targetTumorCellId = null,
                    speed = 2.5f + (float)(_rng.NextDouble() * 1.5)
                };
                Hemocytes.Add(h);
            }
        }

        public void Step(
            float dtHours,
            float drugCytotoxicityPerHour,
            float cd47ShieldDisruptionFactor, // 0.0 to 1.0 (from active drug)
            float cholinergicReflexAntiInflammatoryTone
        )
        {
            float totalLactateProduced = 0.0f;
            int viableTumorCount = 0;
            List<TumorCellAgent> newCells = new List<TumorCellAgent>();

            // 1. Tumor Cell Dynamics
            for (int i = 0; i < TumorCells.Count; i++)
            {
                var cell = TumorCells[i];

                if (cell.state == TumorCellState.Apoptotic ||
                    cell.state == TumorCellState.EngulfedByHemocyte ||
                    cell.state == TumorCellState.Necrotic)
                {
                    continue;
                }

                viableTumorCount++;

                // A. Apply Pharmacological Kill Rate modulated by individual cell resistance
                float effectiveKillRate = drugCytotoxicityPerHour * (1.0f - cell.chemoResistance * 0.70f);
                if (effectiveKillRate > 0.001f)
                {
                    cell.vitality -= effectiveKillRate * dtHours;
                    if (cell.vitality <= 0.0f)
                    {
                        cell.state = TumorCellState.Apoptotic;
                        CumulativeApoptoticCount++;
                        continue;
                    }
                }

                // B. Distance from centroid (Oxygen / Nutrient Gradient)
                float dist = cell.position.DistanceTo(_tumorCentroid);
                if (dist < 15.0f && TumorCells.Count > 100)
                {
                    cell.state = TumorCellState.Hypoxic;
                    cell.timeInHypoxia += dtHours;
                    cell.lactateProduction = 0.22f; // Increased Warburg glycolysis under hypoxia
                }
                else
                {
                    if (cell.state == TumorCellState.Hypoxic)
                        cell.state = TumorCellState.Proliferating;
                }

                totalLactateProduced += cell.lactateProduction * dtHours;

                // C. Proliferation (Gompertzian / logistic growth rate)
                float proliferationChance = (cell.state == TumorCellState.CancerStemCell) ? 0.08f : 0.035f;
                // Suppressed by high drug concentrations
                proliferationChance *= Math.Max(0.02f, 1.0f - drugCytotoxicityPerHour * 3.0f);

                if (_rng.NextDouble() < proliferationChance * dtHours && TumorCells.Count + newCells.Count < 2500)
                {
                    Vector3D divOffset = SampleSphereGaussian(6.0f);
                    newCells.Add(new TumorCellAgent
                    {
                        id = _nextTumorId++,
                        position = new Vector3D(cell.position.x + divOffset.x, cell.position.y + divOffset.y, cell.position.z + divOffset.z),
                        state = TumorCellState.Proliferating,
                        vitality = 1.0f,
                        chemoResistance = Math.Min(0.95f, cell.chemoResistance + (float)(_rng.NextDouble() * 0.05 - 0.02)),
                        cd47Expression = cell.cd47Expression,
                        lactateProduction = 0.08f,
                        timeInHypoxia = 0.0f
                    });
                }
            }

            TumorCells.AddRange(newCells);

            // 2. Hemocyte Immune Surveillance & Phagocytosis
            for (int hIdx = 0; hIdx < Hemocytes.Count; hIdx++)
            {
                var h = Hemocytes[hIdx];
                if (h.phagocyticCapacity <= 0.0f)
                {
                    h.state = HemocyteState.ExhaustedByTumorShield;
                    continue;
                }

                // If primed by neuro-immune acetylcholine tone, speed and sensitivity increase
                float effectiveSpeed = h.speed * (0.7f + cholinergicReflexAntiInflammatoryTone * 0.3f);

                // Find nearest viable tumor cell
                TumorCellAgent target = null;
                float closestDist = 99999f;

                for (int tIdx = 0; tIdx < TumorCells.Count; tIdx++)
                {
                    var tc = TumorCells[tIdx];
                    if (tc.state == TumorCellState.Apoptotic || tc.state == TumorCellState.EngulfedByHemocyte)
                        continue;

                    float d = h.position.DistanceTo(tc.position);
                    if (d < closestDist)
                    {
                        closestDist = d;
                        target = tc;
                    }
                }

                if (target != null)
                {
                    h.targetTumorCellId = target.id;

                    // Chemotactic migration towards target
                    Vector3D dir = new Vector3D(
                        target.position.x - h.position.x,
                        target.position.y - h.position.y,
                        target.position.z - h.position.z
                    );
                    float len = Math.Max(0.001f, closestDist);
                    h.position.x += (dir.x / len) * effectiveSpeed * dtHours * 20.0f;
                    h.position.y += (dir.y / len) * effectiveSpeed * dtHours * 20.0f;
                    h.position.z += (dir.z / len) * effectiveSpeed * dtHours * 20.0f;

                    // Phagocytic engulfment radius: 10 microns
                    if (closestDist <= 10.0f)
                    {
                        // Check CD47 Shield: if shield is breached by drug, engulfment proceeds!
                        float effectiveShield = target.cd47Expression * (1.0f - cd47ShieldDisruptionFactor);
                        if (_rng.NextDouble() > effectiveShield)
                        {
                            target.state = TumorCellState.EngulfedByHemocyte;
                            h.state = HemocyteState.ActivelyEngulfing;
                            h.phagocyticCapacity -= 1.0f;
                            CumulativeApoptoticCount++;
                        }
                        else
                        {
                            // Stymied by 'Don't-Eat-Me' CD47 signaling
                            h.state = HemocyteState.ExhaustedByTumorShield;
                        }
                    }
                    else
                    {
                        h.state = HemocyteState.Patrolling;
                    }
                }
            }

            // 3. Systemic Microenvironment & Cachexia Telemetry
            SystemicLactateMm = Math.Max(1.0f, SystemicLactateMm + totalLactateProduced * 0.05f - 0.02f * dtHours);
            
            // Cachexia is driven by tumor load + lactate acidity, mitigated by cholinergic anti-inflammatory reflex
            float rawCachexiaLoad = (viableTumorCount / 10.0f) + (SystemicLactateMm * 4.5f);
            float reflexProtection = Math.Max(0.3f, 2.5f - cholinergicReflexAntiInflammatoryTone);
            CachexiaScore = Math.Min(100.0f, Math.Max(0.0f, CachexiaScore + (rawCachexiaLoad * reflexProtection * 0.015f * dtHours) - 0.05f * dtHours));
        }

        private Vector3D SampleSphereGaussian(float stdDev)
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = 1.0 - _rng.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);

            double theta = _rng.NextDouble() * 2.0 * Math.PI;
            double phi = Math.Acos(2.0 * _rng.NextDouble() - 1.0);

            float r = (float)(Math.Abs(randStdNormal) * stdDev);
            float x = r * (float)(Math.Sin(phi) * Math.Cos(theta));
            float y = r * (float)(Math.Sin(phi) * Math.Sin(theta));
            float z = r * (float)Math.Cos(phi);

            return new Vector3D(x, y, z);
        }
    }
}
