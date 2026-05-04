from typing import List, Optional
import subprocess
import uuid
from concurrent.futures import ProcessPoolExecutor
import torch
from botorch.test_functions.base import BaseTestProblem
from botorch.utils.torch import BufferDict
from torch import Tensor
import os
import numpy as np
#from opm.simulators import BlackOilSimulator
from opm.io.parser import Parser
from opm.io.ecl import EclFile
from opm.io.ecl import EGrid
from opm.io.ecl import ERst
import matplotlib.pyplot as plt


from opm.io.ecl_state import EclipseState
from opm.io.schedule import Schedule
from opm.io.summary import SummaryConfig
from opm.io.deck import DeckKeyword
from opm.io.ecl import ESmry
import matplotlib.pyplot as plt
import shutil
import string
import pickle
from .problem import DiscreteTestProblem


def evaluate_candidate(args: tuple) -> torch.Tensor:
    candidate, n_inj, n_prod = args
    try:
        run_id = str(uuid.uuid4())
        test_function = JHN(negate=True, n_inj=n_inj, n_prod=n_prod)
        result = test_function.evaluate_true_single(candidate, run_id)
        return result
    except Exception as e:
        print(f"Exception during evaluation: {e}")
        return torch.zeros(1, dtype=torch.float64).view(-1, 1)


class JHN(DiscreteTestProblem):
    def __init__(self, n_inj: int,n_prod: int, noise_std: Optional[float] = None, negate: bool = False) -> None:
        """
        Args:
            n_wells: Number of wells.
            noise_std: Standard deviation of noise.
            negate: If True, negate the output.
        """
        self.num_objectives = 1
        self.n_inj = n_inj
        self.n_prod = n_prod
        self.n_wells = n_inj + n_prod
        # Total dimension is 1 (injection rate) + 2 * n_wells (each well's x and y)
        self.dim =  2 * self.n_wells + 2
        self.num_objectives = 1  # For a single-objective problem
        lb = torch.zeros(self.dim)
        ub = torch.ones(self.dim)

        self._bounds = [(0,1)]*self.dim

        integer_indices: List[int] = list(range(2, self.dim))
        super().__init__(noise_std=noise_std, negate=negate, integer_indices=integer_indices)

    def evaluate_true_single(self, X: Tensor, run_id: Optional[str] = None) -> Tensor:
        # Evaluate the true objective
        X_split = list(torch.split(X, 1, -1))
        if run_id is None:
            run_id = str(uuid.uuid4())  # Generate a unique identifier if not provided
        parser = Parser()
        init = EclFile('test_functions/i.1047/JOHANSEN.INIT')
        porv = init['PORV']
        porv_reshaped = porv.reshape(16, 189, 149)
        johansen = porv_reshaped[[8,10,12],:,:]
        valid_johansen_mask = (johansen>50000).all(axis=0)
        seal_form = porv_reshaped[7:8,:,:]
        valid_seal_form_mask = (seal_form>5).all(axis=0)
        valid_seal_form_mask[130:,:40] = False
        valid_seal_form_mask[:78,45:67] = False
        #also mask axis=0 values lower than 176
        total_mask = valid_johansen_mask & valid_seal_form_mask
        total_mask[:,125:]  = False
        y_indices, x_indices = np.where(total_mask)
        valid_locations = list(zip(x_indices, y_indices))
        valid_locations_tensor = torch.tensor(valid_locations, dtype=torch.float32)

        # Normalize the coordinates to [0, 1]
        min_vals = valid_locations_tensor.min(dim=0).values  # min x and min y
        max_vals = valid_locations_tensor.max(dim=0).values  # max x and max y
        normalized_valid_locations = (valid_locations_tensor - min_vals) / (max_vals - min_vals)

        net = []
        for j in range(len(X_split[0])):
            deck = parser.parse('test_functions/Johansen.DATA')
            #unscale the input from [0,1] to the original range
            gas_inj_max = 2.5e6*self.n_inj
            brine_prod_max = 1e4*self.n_prod
            gas_inj = 2e6 + (gas_inj_max-2e6) * X_split[0][j].item()
            brine_prod = 5e3 + (brine_prod_max-5e3) * X_split[1][j].item()
            injector_coords = []

            for i in range(self.n_inj):
                # Get the normalized values from the corresponding splits.
                x_norm = X_split[2 + 2 * i][j].item()  # normalized x for injector i
                y_norm = X_split[2 + 2 * i + 1][j].item()  # normalized y for injector i

                # Denormalize using: original = min + v * (max - min)
                orig_x = min_vals[0].item() + x_norm * (max_vals[0].item() - min_vals[0].item())
                orig_y = min_vals[1].item() + y_norm * (max_vals[1].item() - min_vals[1].item())
                orig_x = int(round(orig_x)) + 1
                orig_y = int(round(orig_y)) + 1
                injector_coords.append((orig_x, orig_y))

           
            producer_coords = []
            start_idx = 2 + 2 * self.n_inj
            for i in range(self.n_prod):
                x_norm = X_split[start_idx + 2 * i][j].item()
                y_norm = X_split[start_idx + 2 * i + 1][j].item()
                orig_x = min_vals[0].item() + x_norm * (max_vals[0].item() - min_vals[0].item())
                orig_y = min_vals[1].item() + y_norm * (max_vals[1].item() - min_vals[1].item())
                orig_x = int(round(orig_x)) + 1
                orig_y = int(round(orig_y)) + 1
                producer_coords.append((orig_x, orig_y))
            all_coords = injector_coords + producer_coords
            if len(set(all_coords)) < len(all_coords):
                net.append(0)
                continue
            welspecs_str = """WELSPECS\n
             

            """
            if producer_coords != []:
                for i, (x, y) in enumerate(producer_coords):
                    welspecs_str += f"'P{i+1}' 'PRODUCERS' {x} {y} 1* 'OIL' 0.2 /\n"
            
            for i, (x, y) in enumerate(injector_coords):
                welspecs_str += f"'I{i+1}' 'INJECTORS' {x} {y} 1* 'GAS' 0.2 /\n"

            welspecs_str += '/\n'
            deck = parser.parse_string(str(deck)+welspecs_str)

            compdat_str = """COMPDAT\n

            """
            if producer_coords != []:
                for i, (x, y) in enumerate(producer_coords):
                    compdat_str += f"'P{i+1}' {x} {y} 9 13 'OPEN' 0 1* 0.2 /\n"
            
            for i, (x, y) in enumerate(injector_coords):
                compdat_str += f"'I{i+1}' {x} {y} 9 13 'OPEN' 0 1* 0.2 /\n"
            
            compdat_str += '/\n'
            deck = parser.parse_string(str(deck)+compdat_str)
            if producer_coords != []:
                gconprod_str = f"""GCONPROD 
                'FIELD' 'ORAT' 150000 1* 50000 1* 'RATE' 'NO' /
                'PRODUCERS' 'ORAT' {brine_prod} 1* 50000 1* 'RATE' 'NO' /
                /
                """
                deck = parser.parse_string(str(deck)+gconprod_str)
            gconinje_str = f"""GCONINJE
            'FIELD' 'GAS' 'RATE' {gas_inj} 3* 'NO' /
            'INJECTORS' 'GAS' 'RATE' {gas_inj} 3* 'NO' /
            /
            """
            deck = parser.parse_string(str(deck)+gconinje_str)
            wconprod_str = """WCONPROD
            """
            if producer_coords != []:
                for i, _ in enumerate(producer_coords):
                    wconprod_str += f"'P{i+1}' 'OPEN' 'GRUP' 10000.0 10000.0 50000 30000.0 1* 120 /\n"

            wconprod_str += '/\n'
            deck = parser.parse_string(str(deck)+wconprod_str)
            wconinje_str = """WCONINJE
            """

            for i, _ in enumerate(injector_coords):
                wconinje_str += f"'I{i+1}' 'GAS' 'OPEN' 'GRUP' 2500000 1* 335 /\n"
            wconinje_str += '/\n'
            deck = parser.parse_string(str(deck)+wconinje_str)

            first_t_step_str = """
            DATES
            1 'JLY' 2022 /
            1 'JAN' 2023 /
            1 'JLY' 2023 /
            1 'JAN' 2024 /
            1 'JLY' 2024 /
            1 'JAN' 2025 /
            1 'JLY' 2025 /
            1 'JAN' 2026 /
            1 'JLY' 2026 /
            1 'JAN' 2027 /
            1 'JAN' 2028 /
            1 'JAN' 2029 /
            1 'JAN' 2030 /
            1 'JAN' 2031 /
            1 'JAN' 2032 /
            1 'JAN' 2033 /
            1 'JAN' 2034 /
            1 'JAN' 2035 /
            1 'JAN' 2036 /
            1 'JAN' 2037 /
            1 'JAN' 2038 /
            1 'JAN' 2040 /
            1 'JAN' 2042 /
            1 'JAN' 2044 /
            1 'JAN' 2046 /
            1 'JAN' 2048 /
            1 'JAN' 2050 /
            1 'JAN' 2052 /
            1 'JAN' 2054 /
            1 'JAN' 2056 /
            1 'JAN' 2058 /
            1 'JAN' 2060 /
            1 'JAN' 2061 /
            1 'JAN' 2062 /
            1 'JAN' 2063 /
            1 'JAN' 2064 /
            1 'JAN' 2065 /
            1 'JAN' 2066 /
            1 'JAN' 2067 /
            1 'JAN' 2069 /
            1 'JAN' 2071 /
            1 'JAN' 2073 /
            1 'JAN' 2075 /
            1 'JAN' 2077 /
            1 'JAN' 2079 /
            1 'JAN' 2081 /
            1 'JAN' 2083 /
            1 'JAN' 2085 /
            1 'JAN' 2087 /
            1 'JAN' 2089 /
            1 'JAN' 2091 /
            1 'JAN' 2093 /
            1 'JAN' 2095 /
            1 'JAN' 2097 /
            1 'JAN' 2099 /
            1 'JAN' 2101 /
            1 'JAN' 2102 /
            /
            """
            deck = parser.parse_string(str(deck)+first_t_step_str)
            shut_wells_str = """WELOPEN
            """
            if producer_coords != []:
                for i, _ in enumerate(producer_coords):
                    shut_wells_str += f"'P{i+1}' 'SHUT' /\n"
            for i, _ in enumerate(injector_coords):
                shut_wells_str += f"'I{i+1}' 'SHUT' /\n"
            shut_wells_str += '/\n'
            deck = parser.parse_string(str(deck)+shut_wells_str)

            second_t_step_str = """
                        DATES
            1 'JAN' 2112 /
            1 'JAN' 2122 /
            1 'JAN' 2132 /
            1 'JAN' 2142 /
            1 'JAN' 2152 /
            1 'JAN' 2162 /
            /
            END
            """
            deck = parser.parse_string(str(deck)+second_t_step_str)
            deck_filename = f'DECK_{run_id}.DATA'
            with open(deck_filename, 'w') as f:
                f.write(str(deck))
            outdir = f'i.{run_id}'

            try:
                subprocess.run(["mpirun", "-np", "16", "--allow-run-as-root","flow", deck_filename, f"--output-dir={outdir}"], check=True,timeout=80*60,
                                       stdout=subprocess.DEVNULL, 
                                        stderr=subprocess.DEVNULL   )
            except subprocess.TimeoutExpired:
                print(f"Process timed out after 20 minutes for run_id {run_id}.")
            except Exception as e:
                print(f"Subprocess run failed for run_id {run_id}: {e}")
                pass
            try:
                caps_id = run_id.upper()
                summary = ESmry(f'{outdir}/DECK_{caps_id}.SMSPEC')
                co2_inflow = 30 * 0.0019   # 30 euro/tonne * tonne/m^3 * m^3/Mscf
                co2_outflow = 6.2 * 0.0019  # 6.2 euro/tonne ...
                brine_treat = 4.83 * 1.2  # 4.83 euro/tonne*tonne/m^3 *m^3/stb
                times = np.array(summary['TIME'])   # e.g. [0, 90, 180, 360, ...]
                times_rounded = np.round(times).astype(int)  # ensure integer days
                FGPR = np.array(summary['FGPR'])  # shape [N+1]
                FGIR = np.array(summary['FGIR'])  # shape [N+1]
                FOPR = np.zeros_like(FGPR)  # shape [N+1]
                if producer_coords != []:
                    FOPR += np.array(summary['GOPR:PRODUCER'])  # shape [N+1]


                # intervals between consecutive saved times
                intervals = times_rounded[1:] - times_rounded[:-1]  # shape [N]

                # For each step i, we replicate FOPR[i] for intervals[i] days.
                daily_FOPR = np.repeat(FOPR[:-1], intervals)  # shape = sum of intervals
                daily_FGPR = np.repeat(FGPR[:-1], intervals)
                daily_FGIR = np.repeat(FGIR[:-1], intervals)

                # Construct a daily array of days from times_rounded[0] to times_rounded[-1]-1
                start_day = times_rounded[0]
                end_day = times_rounded[-1]
                days = np.arange(start_day, end_day)  # shape = sum(intervals)
                inflation_rate = 0.0142  # 1.42%

                # Convert annual inflation rate to daily inflation rate
                daily_discount_rate = (1 + inflation_rate) ** (1 / 365) - 1

                # Calculate daily cash flows
                co2_inflow_cash_flows = (daily_FGIR - daily_FGPR) * co2_inflow
                co2_outflow_cash_flows = daily_FGIR * co2_outflow
                brine_treatment_cash_flows = daily_FOPR * brine_treat

                # Apply inflation adjustment to cash flows
                co2_inflow_present_values = co2_inflow_cash_flows / (1 + daily_discount_rate) ** days
                co2_outflow_present_values = co2_outflow_cash_flows / (1 + daily_discount_rate) ** days
                brine_treatment_present_values = brine_treatment_cash_flows / (1 + daily_discount_rate) ** days

                # Calculate NPV
                npv = np.sum(co2_inflow_present_values) - np.sum(co2_outflow_present_values) - np.sum(brine_treatment_present_values)
                leaked_co2 = summary['FGIP'][-1] -  summary['RGIP:1'][-1]
                saved_co2 = summary['RGIP:1'][-1]

                if leaked_co2 >0:
                    penalty = min(1000*leaked_co2/saved_co2,1)
                    

                    npv  -= npv*penalty 

                if npv > 0:

                    condition = np.logical_and(
                        np.abs(np.diff(summary['FGIR'][20:])) < 0.3 * gas_inj,
                        np.abs(np.diff(summary['FGIR'][20:])) > 0.1 * gas_inj
                    )
                    if condition.any():
                        first_idx = np.nonzero(condition)[0][0]  # first index where condition holds
                        # Use a factor (e.g., 10) to scale the penalty; ensure first_idx is not zero:
                        penalty_factor = (first_idx + 1)/100 
                        npv -= npv / penalty_factor

            except Exception as e:
                # Print exception details to understand what went wrong
                print("Exception occurred:", e)
                npv=0
                net.append(npv)
                shutil.rmtree(outdir, ignore_errors=True)
                os.remove(deck_filename)
                return torch.zeros(1, dtype=torch.float64).view(-1, 1)
            net.append(npv/1e11)

            shutil.rmtree(outdir, ignore_errors=True)
            os.remove(deck_filename)

        return -torch.tensor(net, dtype=torch.float64).view(-1,1)
    

    def evaluate_true_parallel(self, X: Tensor) -> Tensor:
        n_candidates = X.shape[0]
        # Include n_inj and n_prod with each candidate
        args_list = [(X[i], self.n_inj, self.n_prod) for i in range(n_candidates)]
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(evaluate_candidate, args_list))
        results_tensor = torch.cat(results, dim=0)  # (n, 1)
        # results_tensor = results_tensor.squeeze()  # now shape is (n_candidates, 1)

        return results_tensor

    # Override the evaluate_true method to use parallel evaluation
    def evaluate_true(self, X: Tensor, run_id: Optional[str] = None) -> Tensor:
        """
        Override the evaluate_true method to perform parallel evaluations.

        Args:
            X (Tensor): Tensor of candidates with shape [n_candidates, dim].
            run_id (Optional[str]): Unique identifier for the evaluation run.

        Returns:
            Tensor: Evaluation results with shape [n_candidates, num_objectives].
        """
        return self.evaluate_true_parallel(X)