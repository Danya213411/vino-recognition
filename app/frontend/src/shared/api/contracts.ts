export type Decision = "match" | "alternatives" | "not_found";

export type Wine = {
  slug: string;
  title: string;
  manufacturer?: string | null;
  region?: string | null;
  category?: string | null;
  color?: string | null;
  description?: string | null;
  grapes?: string[] | null;
  dishes?: string[] | null;
  alcohol?: number | null;
  rating?: number | null;
  public_rating?: number | null;
  temperature?: string | null;
  source_url?: string | null;
  color_gradient?: string | null;
  image_url: string;
};

export type Prediction = {
  slug: string;
  score: number;
  embedding_score: number;
  sift_score: number;
  sift_inliers: number;
  ocr_score: number;
  relative_support: number;
  candidate_sources?: string[];
  wine: Wine;
};

export type Analysis = {
  id: string;
  created_at: string;
  mode: "user" | "calib";
  decision: Decision;
  slug: string | null;
  candidate_slug: string;
  confidence: number;
  calibrated_confidence: number;
  confidence_source: "calibrated_model" | "multimodal_verification";
  verified_match: boolean;
  thresholds: { accept: number; review: number };
  wine: Wine;
  predictions: Prediction[];
  evidence: Record<string, number>;
  ocr_lines: Array<{ text?: string; normalized?: string; score?: number }>;
  manufacturer_match?: {
    value: string | null;
    values?: string[];
    score: number;
    runner_up: string | null;
    runner_score: number;
    margin: number;
    mode: "hard" | "none";
    matched_text: string | null;
    candidate_count: number;
  } | null;
  crop_consistency: Record<string, number | string[]>;
  timing_ms: Record<string, number>;
  memory: Record<string, number>;
  manifest_url: string;
};

export type FeedbackVerdict = "correct" | "incorrect" | "not_in_catalog" | "not_in_store";

export type ReviewSetItem = {
  name: string;
  size: number;
  image_url: string;
};

export type AdminStats = {
  recognitions: {
    total: number;
    matches: number;
    alternatives: number;
    not_found: number;
  };
  timing_ms: { mean: number | null; p50: number | null; p95: number | null };
  feedback: {
    total: number;
    correct: number;
    incorrect: number;
    not_in_catalog: number;
    not_in_store: number;
  };
};

export type AdminRecord = {
  id: string;
  created_at: string;
  session_id: string;
  mode: "user" | "calib";
  original_name: string;
  width: number;
  height: number;
  byte_size: number;
  decision: Decision;
  predicted_slug: string | null;
  candidate_slug: string;
  confidence: number;
  total_ms: number;
  result: Analysis;
  feedback: null | {
    verdict: FeedbackVerdict;
    correct_slug?: string | null;
    note?: string | null;
  };
  pipeline: Record<string, unknown>;
  image_path: string;
  mime_type: string;
  client_host?: string | null;
  user_agent?: string | null;
};

export type ReviewMatrixItem = {
  name: string;
  input_image_url: string;
  recognition: AdminRecord;
};
