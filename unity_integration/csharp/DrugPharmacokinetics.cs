// ============================================================================
// DROSOPHILA IN SILICO DIGITAL TWIN - UNITY 3D / C# .NET ENGINE
// File: DrugPharmacokinetics.cs
// Namespace: DrosophilaTwin.Core
// Author: Muhammed Emre Albayrak
// License: MIT
// ============================================================================

using System;

namespace DrosophilaTwin.Core
{
    public class DrugPharmacokinetics
    {
        public string ActiveCompoundName { get; private set; } = "DeNovo_Champion";
        public float PlasmaConcentrationUm { get; private set; } = 0.0f;
        public float BrainConcentrationUm { get; private set; } = 0.0f;
        public float ReceptorOccupancyRatio { get; private set; } = 0.0f;
        public float TargetInhibitionPct { get; private set; } = 0.0f;
        public float CytotoxicRatePerHour { get; private set; } = 0.0f;
        public float Cd47ShieldBreachFactor { get; private set; } = 0.0f;

        // Pharmacological Parameters
        private float _halfLifeHours = 8.5f;
        private float _bloodBrainPermeability = 0.65f;
        private float _affinityKdUm = 0.045f; // 45 nM
        private float _maxEfficacy = 0.95f;
        private float _hillCoefficient = 1.4f;

        public DrugPharmacokinetics(string compoundName = "DeNovo_Champion")
        {
            SetCompound(compoundName);
        }

        public void SetCompound(string name)
        {
            ActiveCompoundName = name;
            string lower = name.ToLower();

            if (lower.Contains("denovo") || lower.Contains("f-nic"))
            {
                _halfLifeHours = 10.5f;
                _bloodBrainPermeability = 0.82f; // High BBB penetration (soft-drug)
                _affinityKdUm = 0.038f;          // 38 nM
                _maxEfficacy = 0.94f;
                _hillCoefficient = 1.5f;
            }
            else if (lower.Contains("cisplatin"))
            {
                _halfLifeHours = 4.2f;
                _bloodBrainPermeability = 0.22f; // Low insect BBB penetration
                _affinityKdUm = 1.20f;           // 1.2 µM
                _maxEfficacy = 0.78f;
                _hillCoefficient = 1.0f;
            }
            else if (lower.Contains("trametinib"))
            {
                _halfLifeHours = 72.0f;
                _bloodBrainPermeability = 0.45f;
                _affinityKdUm = 0.0012f;         // 1.2 nM (Ultra-potent MEK)
                _maxEfficacy = 0.92f;
                _hillCoefficient = 1.8f;
            }
            else if (lower.Contains("cocktail") || lower.Contains("multimodal"))
            {
                _halfLifeHours = 12.0f;
                _bloodBrainPermeability = 0.88f;
                _affinityKdUm = 0.024f;          // Synergy cocktail
                _maxEfficacy = 0.98f;
                _hillCoefficient = 2.0f;
            }
            else
            {
                _halfLifeHours = 6.0f;
                _bloodBrainPermeability = 0.50f;
                _affinityKdUm = 0.25f;
                _maxEfficacy = 0.80f;
                _hillCoefficient = 1.2f;
            }
        }

        public void AdministerBolusDose(float doseUm)
        {
            PlasmaConcentrationUm += doseUm;
        }

        public void Step(float dtHours)
        {
            if (dtHours <= 0.0f) return;

            // First-order elimination from plasma: C(t) = C0 * exp(-k_el * t)
            float kEl = (float)(Math.Log(2.0) / _halfLifeHours);
            PlasmaConcentrationUm = Math.Max(0.0f, PlasmaConcentrationUm * (float)Math.Exp(-kEl * dtHours));

            // Interstitial Brain Tissue Equilibration
            float targetBrain = PlasmaConcentrationUm * _bloodBrainPermeability;
            BrainConcentrationUm += (targetBrain - BrainConcentrationUm) * (dtHours * 1.8f);
            BrainConcentrationUm = Math.Max(0.0f, BrainConcentrationUm);

            // Hill Receptor Occupancy Equation: theta = C^h / (C^h + Kd^h)
            if (BrainConcentrationUm > 0.0001f)
            {
                double cPow = Math.Pow(BrainConcentrationUm, _hillCoefficient);
                double kdPow = Math.Pow(_affinityKdUm, _hillCoefficient);
                ReceptorOccupancyRatio = (float)(cPow / (cPow + kdPow));
            }
            else
            {
                ReceptorOccupancyRatio = 0.0f;
            }

            TargetInhibitionPct = ReceptorOccupancyRatio * _maxEfficacy * 100.0f;

            // Efficacy Translation to Cell Death Rate
            // DeNovo and Multimodal cocktails induce high targeted cytotoxicity with low off-target toxicity
            CytotoxicRatePerHour = ReceptorOccupancyRatio * _maxEfficacy * 0.45f;

            // If compound is DeNovo or Cocktail, trigger CD47 macrophage checkpoint breach
            if (ActiveCompoundName.ToLower().Contains("denovo") || ActiveCompoundName.ToLower().Contains("cocktail"))
            {
                Cd47ShieldBreachFactor = ReceptorOccupancyRatio * 0.85f;
            }
            else
            {
                Cd47ShieldBreachFactor = ReceptorOccupancyRatio * 0.25f;
            }
        }

        public PharmacokineticTelemetry GetTelemetry()
        {
            return new PharmacokineticTelemetry
            {
                compoundName = ActiveCompoundName,
                plasmaConcentrationUm = PlasmaConcentrationUm,
                brainIntercellularConcentrationUm = BrainConcentrationUm,
                receptorOccupancyRatio = ReceptorOccupancyRatio,
                targetKinaseInhibitionPct = TargetInhibitionPct,
                cytotoxicRatePerHour = CytotoxicRatePerHour
            };
        }
    }
}
