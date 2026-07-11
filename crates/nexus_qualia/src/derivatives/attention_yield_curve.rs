//! Attention Yield Curve - maps cognitive load to attention-based returns.
//! Similar to bond yield curves but for digital platform engagement.

/// Maximum maturities supported
pub const MAX_MATURITIES: usize = 12;

/// Attention yield curve point
#[derive(Debug, Clone)]
pub struct YieldCurvePoint {
    pub maturity_months: u32,
    pub attention_yield: f32,
    pub cognitive_load: f32,
    pub churn_risk: f32,
}

impl YieldCurvePoint {
    pub const fn new() -> Self {
        Self {
            maturity_months: 0,
            attention_yield: 0.0,
            cognitive_load: 0.0,
            churn_risk: 0.0,
        }
    }
}

impl Default for YieldCurvePoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Attention yield curve result
#[derive(Debug, Clone)]
pub struct AttentionYieldCurve {
    pub points: [YieldCurvePoint; MAX_MATURITIES],
    pub num_points: usize,
    pub slope: f32,
    pub is_inverted: bool,
    pub area_under_curve: f32,
}

impl AttentionYieldCurve {
    pub const fn new() -> Self {
        Self {
            points: [YieldCurvePoint::new(); MAX_MATURITIES],
            num_points: 0,
            slope: 0.0,
            is_inverted: false,
            area_under_curve: 0.0,
        }
    }
}

impl Default for AttentionYieldCurve {
    fn default() -> Self {
        Self::new()
    }
}

/// Main attention yield curve engine
pub struct AttentionYieldCurveBuilder {
    curve: AttentionYieldCurve,
    base_cognitive_load: f32,
    load_sensitivity: f32,
}

impl AttentionYieldCurveBuilder {
    pub fn new() -> Self {
        Self {
            curve: AttentionYieldCurve::new(),
            base_cognitive_load: 50.0,
            load_sensitivity: 0.02,
        }
    }

    /// Configure parameters
    pub fn configure(&mut self, base_load: f32, sensitivity: f32) {
        self.base_cognitive_load = base_load.clamp(0.0, 100.0);
        self.load_sensitivity = sensitivity.max(0.0);
    }

    /// Build yield curve from cognitive load data
    pub fn build(&mut self, cognitive_loads: &[f32; MAX_MATURITIES]) -> &AttentionYieldCurve {
        self.curve.num_points = MAX_MATURITIES;
        let maturities = [1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 84, 120];

        for i in 0..MAX_MATURITIES {
            self.curve.points[i].maturity_months = maturities[i];
            self.curve.points[i].cognitive_load = cognitive_loads[i];
            
            // Yield = base_yield + load_premium - fatigue_discount
            let load_premium = (cognitive_loads[i] - self.base_cognitive_load) * self.load_sensitivity;
            let fatigue_discount = cognitive_loads[i].powf(2.0) * 0.0001;
            self.curve.points[i].attention_yield = (0.03 + load_premium - fatigue_discount).max(0.0);
            
            // Churn risk increases with cognitive load
            self.curve.points[i].churn_risk = (cognitive_loads[i] / 100.0).powf(3.0);
        }

        // Calculate slope (short vs long term)
        if self.curve.num_points >= 2 {
            self.curve.slope = self.curve.points[MAX_MATURITIES - 1].attention_yield 
                - self.curve.points[0].attention_yield;
            self.curve.is_inverted = self.curve.slope < 0.0;
        }

        // Area under curve (trapezoidal)
        self.curve.area_under_curve = self.compute_area();

        &self.curve
    }

    fn compute_area(&self) -> f32 {
        let mut area = 0.0f32;
        for i in 1..self.curve.num_points {
            let dt = (self.curve.points[i].maturity_months - 
                     self.curve.points[i-1].maturity_months) as f32;
            let avg_yield = (self.curve.points[i].attention_yield + 
                            self.curve.points[i-1].attention_yield) / 2.0;
            area += dt * avg_yield;
        }
        area / 12.0 // Annualize
    }

    /// Get yield at specific maturity
    pub fn get_yield(&self, maturity_months: u32) -> Option<f32> {
        for point in &self.curve.points[..self.curve.num_points] {
            if point.maturity_months == maturity_months {
                return Some(point.attention_yield);
            }
        }
        None
    }

    /// Get curve slope
    #[inline]
    pub const fn slope(&self) -> f32 {
        self.curve.slope
    }

    /// Check if inverted
    #[inline]
    pub const fn is_inverted(&self) -> bool {
        self.curve.is_inverted
    }
}

impl Default for AttentionYieldCurveBuilder {
    fn default() -> Self {
        Self::new()
    }
}
