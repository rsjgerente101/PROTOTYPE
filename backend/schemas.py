from typing import List, Optional
from pydantic import BaseModel


class BaselineRequest(BaseModel):
    dataset_id: str
    num_representatives: int = 4
    avg_speed_kmph: float = 40.0
    service_minutes_per_stop: float = 8.0
    seed: int = 42
    run_profile: Optional[str] = "default_balanced"


class EnhancedRequest(BaseModel):
    dataset_id: str
    baseline_run_id: str
    alpha_weight: Optional[float] = None
    beta_weight: Optional[float] = None
    max_iterations: Optional[int] = None
    border_fraction: Optional[float] = None
    run_profile: Optional[str] = None


class AddedCustomerPayload(BaseModel):
    label: str
    lat: float
    lon: float
    address: Optional[str] = None
    assigned_rep: Optional[str] = None
    customer_number: Optional[int] = None


class BaselineAddCustomersRequest(BaseModel):
    baseline_run_id: str
    customers: List[AddedCustomerPayload]
