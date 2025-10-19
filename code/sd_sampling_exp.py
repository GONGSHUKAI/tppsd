import torch
import dpp
from dpp.data import Batch
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from scipy import stats 
import yaml
import argparse
import os
from utils import load_models, plot_sampling_results, plot_ll_comparison, add_synth_dataset, plot_temporal_goodness_of_fit
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class SpeculativeTPPSampler:
    """
    Implements speculative decoding for temporal point process sampling.
    Uses a smaller draft model to generate candidate samples that are verified
    by a larger target model, potentially generating multiple events in parallel.
    """
    def __init__(
        self, 
        target_model:dpp.models.LogNormMixTransformer,
        draft_model:dpp.models.LogNormMixTransformer,
        gamma=5,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma
        self.device = device
        
        # Move models to the appropriate device
        self.target_model.to(self.device)
        self.draft_model.to(self.device)
        
        # Set models to evaluation mode
        self.target_model.eval()
        self.draft_model.eval()
    
    def sample(self, init_batch, batch_size=1, t_end=100, max_events=1000):
        """
        Sample multiple point processes using speculative decoding, initialized with the same history.
        
        Args:
            init_batch: Batch object containing initial historical event data
            batch_size: Number of independent point processes to simulate
            t_end: Upper time bound for the simulation
            max_events: Maximum number of events to generate per process
            
        Returns:
            final_batch: Batch object with generated events for all processes
            metrics: Dictionary with sampling metrics
        """
        # Start with minimal history - just the last event time and mark from init_batch
        last_time = init_batch.event_times[:, -1:]
        event_times = last_time.expand(batch_size, -1)
        mask = torch.ones_like(event_times)
        
        if init_batch.marks is not None:
            marks = init_batch.marks[:, -1:].expand(batch_size, -1)
        else:
            marks = None
        
        # Create a batch with just the last event
        if marks is not None:
            current_batch = Batch(
                inter_times=torch.ones_like(event_times),
                event_times=event_times,
                mask=mask,
                marks=marks,
                t_ends=torch.full((batch_size,), t_end, device=self.device)
            )
        else:
            current_batch = Batch(
                inter_times=torch.ones_like(event_times),
                event_times=event_times,
                mask=mask,
                t_ends=torch.full((batch_size,), t_end, device=self.device)
            )
        
        # Metrics for tracking performance
        num_iterations = 0
        total_events_generated = 0
        total_events_drafted = 0
        total_accepted_drafts = 0
        start_time = time.time()
        
        while current_batch.event_times.shape[1] < max_events + 1:  # +1 for the initial event
            num_iterations += 1
            
            # Run one step of speculative decoding
            new_batch, avg_accepted = self.sample_step(current_batch, t_end)
            # Break if no new events were generated (e.g., reached t_end)
            if new_batch is None:
                break
                
            # Update current batch with newly generated events
            num_new_events = new_batch.event_times.shape[1]
            
            # Combine historical events with new events
            combined_event_times = torch.cat([current_batch.event_times, new_batch.event_times], dim=1)
            combined_mask = torch.cat([current_batch.mask, new_batch.mask], dim=1)
            
            # Calculate inter-event times from event times
            combined_inter_times = torch.ones_like(combined_event_times)
            combined_inter_times[:, 1:] = torch.diff(combined_event_times, dim=1)
            
            if marks is not None:
                combined_marks = torch.cat([current_batch.marks, new_batch.marks], dim=1)
                current_batch = Batch(
                    inter_times=combined_inter_times,
                    event_times=combined_event_times,
                    mask=combined_mask,
                    marks=combined_marks,
                    t_ends=current_batch.t_ends
                )
            else:
                current_batch = Batch(
                    inter_times=combined_inter_times,
                    event_times=combined_event_times,
                    mask=combined_mask,
                    t_ends=current_batch.t_ends
                )
                
            # Update metrics
            total_events_generated += num_new_events
            total_events_drafted += self.gamma * batch_size
            total_accepted_drafts += avg_accepted * batch_size
            # Check if we've reached t_end
            if (current_batch.event_times[:, -1] >= t_end).any():
                # Truncate any events past t_end
                valid_mask = current_batch.event_times <= t_end
                
                # Find the last valid index for each batch item
                last_valid = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                for batch_idx in range(batch_size):
                    valid_indices = valid_mask[batch_idx].nonzero(as_tuple=True)[0]
                    if len(valid_indices) > 0:
                        last_valid[batch_idx] = valid_indices[-1]
                
                max_valid_len = last_valid.max().item() + 1
                
                # Truncate the batch
                current_batch.event_times = current_batch.event_times[:, :max_valid_len]
                current_batch.mask = current_batch.mask[:, :max_valid_len]
                current_batch.inter_times = current_batch.inter_times[:, :max_valid_len]
                if marks is not None:
                    current_batch.marks = current_batch.marks[:, :max_valid_len]
                
                break
                
        # Calculate final metrics
        sampling_time = time.time() - start_time
        speedup_factor = total_events_generated / (num_iterations * batch_size)
        acceptance_rate = total_accepted_drafts / (total_events_drafted * batch_size)
        
        metrics = {
            "iterations": num_iterations,
            "events_generated": total_events_generated,
            "acceptance_rate": acceptance_rate.item() if isinstance(acceptance_rate, torch.Tensor) else acceptance_rate,
            "sampling_time": sampling_time,
            "events_per_iteration": speedup_factor
        }
        
        return current_batch, metrics
    
    def sample_step(self, batch:dpp.data.Batch, t_end=100):
        """
        Perform one step of speculative decoding to sample multiple future events.
        
        Args:
            batch: Batch object containing current event data
            t_end: Upper time bound for the simulation
            
        Returns:
            new_batch: Batch object containing newly generated events
            num_accepted: Average number of draft model samples that were accepted
        """
        batch_size = batch.size
        
        # Step 1: Use draft model to generate gamma guesses autoregressively
        draft_inter_times = []
        draft_marks = [] if batch.marks is not None else None
        draft_probs = []
        draft_dists = []
        
        current_batch = batch

        # start_time = time.time()
        for i in range(self.gamma):
            # Get context from draft model
            with torch.no_grad():
                context = self.draft_model.get_context(current_batch)[:, -1:, :]
                
                # Sample inter-event time from draft model
                inter_time_dist = self.draft_model.get_inter_time_dist(context)
                draft_dists.append(inter_time_dist)
                next_inter_time = inter_time_dist.sample()  # [batch_size, 1]
                draft_inter_times.append(next_inter_time)
                
                # Calculate probability of the sampled inter-time under draft model
                prob = inter_time_dist.log_prob(next_inter_time).exp()
                draft_probs.append(prob)
                
                # Sample mark if needed
                if batch.marks is not None:
                    mark_logits = torch.log_softmax(self.draft_model.mark_linear(context), dim=-1)
                    mark_dist = torch.distributions.Categorical(logits=mark_logits)
                    next_mark = mark_dist.sample()  # [batch_size, 1]
                    draft_marks.append(next_mark)
                
                # Update current_batch for next iteration by adding this new event. Check if we've exceeded t_end
                next_time = current_batch.event_times[:, -1:] + next_inter_time
                if (next_time >= t_end).any():
                    break
                
                new_event_times = torch.cat([current_batch.event_times, next_time], dim=1)
                new_mask = torch.cat([current_batch.mask, torch.ones_like(next_time)], dim=1)
                new_inter_times = torch.cat([current_batch.inter_times, next_inter_time], dim=1)
                # # Calculate inter-event times from event times
                # new_inter_times = torch.ones_like(new_event_times)
                # new_inter_times[:, 1:] = torch.diff(new_event_times, dim=1)
                
                if batch.marks is not None:
                    new_marks = torch.cat([current_batch.marks, next_mark], dim=1)
                    current_batch = Batch(
                        inter_times=new_inter_times,
                        event_times=new_event_times,
                        mask=new_mask,
                        marks=new_marks,
                        t_ends=current_batch.t_ends
                    )
                else:
                    current_batch = Batch(
                        inter_times=new_inter_times,
                        event_times=new_event_times,
                        mask=new_mask,
                        t_ends=current_batch.t_ends
                    )

        # print(f'time for sampling {self.gamma} events from draft model autoregressively : {time.time() - start_time:.5f}')
        
        # If we didn't generate any draft samples, return None
        if len(draft_inter_times) == 0:
            return None, 0
            
        # Convert lists to tensors
        draft_inter_times = torch.cat(draft_inter_times, dim=1)  # [batch_size, gamma]
        if batch.marks is not None:
            draft_marks = torch.cat(draft_marks, dim=1)  # [batch_size, gamma]
        draft_probs = torch.cat(draft_probs, dim=1)  # [batch_size, gamma]
        
        # Step 2: Use target model to evaluate all guesses in parallel
        # First, prepare the batch with all draft guesses
        start_time = time.time()
        all_event_times = batch.event_times.clone()
        all_mask = batch.mask.clone()
        all_inter_times = batch.inter_times.clone()
        
        # Get the last event time from the current batch and compute all future times at once
        future_event_times = all_event_times[:, -1:] + torch.cumsum(draft_inter_times, dim=1)  # Shape: [batch_size, gamma]
        
        all_event_times = torch.cat([all_event_times, future_event_times], dim=1)  # Shape: [batch_size, orig_len + gamma]
        all_mask = torch.cat([all_mask, torch.ones_like(future_event_times)], dim=1)  # Shape: [batch_size, orig_len + gamma]
        all_inter_times = torch.cat([all_inter_times, draft_inter_times], dim=1)  # Shape: [batch_size, orig_len + gamma]

        if batch.marks is not None:
            all_marks = batch.marks.clone()
            all_marks = torch.cat([all_marks, draft_marks], dim=1)  # Shape: [batch_size, orig_len + gamma]

        # Create batch with all events for parallel evaluation
        if batch.marks is not None:
            all_batch = Batch(
                inter_times=all_inter_times,
                event_times=all_event_times,
                mask=all_mask,
                marks=all_marks,
                t_ends=batch.t_ends
            )
        else:
            all_batch = Batch(
                inter_times=all_inter_times,
                event_times=all_event_times,
                mask=all_mask,
                t_ends=batch.t_ends
            )

        # Get contexts for target model
        # start_time = time.time()
        with torch.no_grad():
            all_contexts = self.target_model.get_context(all_batch, remove_last=False)
            start_idx = batch.event_times.shape[1] - 1
            end_idx = start_idx + draft_inter_times.shape[1]
            relevant_contexts = all_contexts[:, start_idx:end_idx, :]  # Shape: [batch_size, gamma, context_size]
            
            inter_time_dist = self.target_model.get_inter_time_dist(relevant_contexts)
            # Get log probabilities and convert to probabilities
            target_probs = inter_time_dist.log_prob(draft_inter_times).exp()  # Shape: [1, gamma]
        # print(f'time for target model verification: {time.time() - start_time:.5f}')

        # Step 3: Determine the number of accepted guesses
        start_time = time.time()
        acceptance_ratios = torch.min(target_probs / draft_probs, torch.ones_like(target_probs))
        rand_values = torch.rand_like(acceptance_ratios)
        
        # Find the first guess that gets rejected for each batch item
        rejected = (rand_values > acceptance_ratios).float()
        
        # # If all guesses are accepted, set n to gamma
        # n = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # for batch_idx in range(batch_size):
        #     # Find the first rejection index
        #     rejection_indices = (rejected[batch_idx] == 1).nonzero(as_tuple=True)[0]
        #     if len(rejection_indices) > 0:
        #         n[batch_idx] = rejection_indices[0]
        #     else:
        #         n[batch_idx] = draft_inter_times.shape[1]

        # cancel out the batchsize dimension
        rejection_indices = (rejected == 1).nonzero(as_tuple=True)[0]
        if len(rejection_indices) > 0:
            n = rejection_indices[0]
        else:
            n = draft_inter_times.shape[1]
        
        print(f'check rejection idx: {n}')
        # Step 4: For the first rejected index, sample from adjusted distribution
        if n > 0:
            result_inter_times = draft_inter_times[0, :n]
            if n < draft_inter_times.shape[1]:
                context_idx = batch.event_times.shape[1] + n - 1
                context = all_contexts[:, context_idx:context_idx+1, :]

                target_dist = self.target_model.get_inter_time_dist(context)
                draft_dist = draft_dists[n]
                next_inter_time = self.rejection_sampling(target_dist, draft_dist)
                # next_inter_time = self.stupid_sampling(target_dist, draft_dist)
                result_inter_times = torch.cat([result_inter_times, next_inter_time.unsqueeze(0)])
                if batch.marks is not None:
                    mark_logits = torch.log_softmax(self.target_model.mark_linear(context), dim=-1)
                    mark_dist = torch.distributions.Categorical(logits=mark_logits)
                    next_mark = mark_dist.sample()
            
            result_inter_times = result_inter_times.unsqueeze(0)
            
            last_time = batch.event_times[0, -1]
            result_event_times = last_time + torch.cumsum(result_inter_times, dim=1)
            mask = torch.ones_like(result_event_times)
            if batch.marks is not None:
                if n > 0:
                    result_marks = draft_marks[0, :n]
                    
                    if n < draft_inter_times.shape[1]:
                        result_marks = torch.cat([result_marks, next_mark])
                        
                    result_marks = result_marks.unsqueeze(0)
                else:
                    # Only have the sampled mark from target model
                    result_marks = next_mark.unsqueeze(0).unsqueeze(0)
                    
                new_batch = Batch(
                    inter_times=result_inter_times,
                    event_times=result_event_times,
                    mask=mask,
                    marks=result_marks,
                    t_ends=batch.t_ends
                )
            else:
                new_batch = Batch(
                    inter_times=result_inter_times,
                    event_times=result_event_times,
                    mask=mask,
                    t_ends=batch.t_ends
                )
        elif n == 0 and draft_inter_times.shape[1] > 0:
            # All draft predictions were rejected, sample one from target model
            context_idx = batch.event_times.shape[1] - 1
            context = all_contexts[:, context_idx:context_idx+1, :]
            
            target_dist = self.target_model.get_inter_time_dist(context)
            draft_dist = draft_dists[0]
            next_inter_time = self.rejection_sampling(target_dist, draft_dist)
            # next_inter_time = self.stupid_sampling(target_dist, draft_dist)
            result_inter_times = next_inter_time 
            last_time = batch.event_times[0, -1]
            result_event_times = last_time + next_inter_time
            mask = torch.ones_like(result_event_times)
            
            if batch.marks is not None:
                mark_logits = torch.log_softmax(self.target_model.mark_linear(context), dim=-1)
                mark_dist = torch.distributions.Categorical(logits=mark_logits)
                next_mark = mark_dist.sample()
                new_batch = Batch(
                    inter_times=result_inter_times,
                    event_times=result_event_times,
                    mask=mask,
                    marks=next_mark,
                    t_ends=batch.t_ends
                )
            else:
                new_batch = Batch(
                    inter_times=result_inter_times,
                    event_times=result_event_times,
                    mask=mask,
                    t_ends=batch.t_ends
                )
        else:
            # No events were generated
            new_batch = None
        # print(f'time for rejection sampling: {time.time() - start_time:.5f}')
        return new_batch, n
    
    def standard_sample(self, init_batch, batch_size=1, t_end=100, max_events=1000):
        """
        Standard autoregressive sampling with the target model.
        
        Args:
            init_batch: Batch object containing initial historical event data
            batch_size: Number of independent point processes to simulate
            t_end: Upper time bound for the simulation
            max_events: Maximum number of events to generate per process
            
        Returns:
            batch: Batch object with all events
        """
        # Start with minimal history - just the last event time and mark from init_batch
        last_time = init_batch.event_times[:, -1:]
        event_times = last_time.expand(batch_size, -1)
        mask = torch.ones_like(event_times)
        
        if init_batch.marks is not None:
            marks = init_batch.marks[:, -1:].expand(batch_size, -1)
        else:
            marks = None
        
        # Create a batch with just the last event
        if marks is not None:
            current_batch = Batch(
                inter_times=torch.ones_like(event_times),
                event_times=event_times,
                mask=mask,
                marks=marks,
                t_ends=torch.full((batch_size,), t_end, device=self.device)
            )
        else:
            current_batch = Batch(
                inter_times=torch.ones_like(event_times),
                event_times=event_times,
                mask=mask,
                t_ends=torch.full((batch_size,), t_end, device=self.device)
            )

        num_generated = 0
        
        while num_generated < max_events:
            # start_time = time.time()
            with torch.no_grad():
                context = self.target_model.get_context(current_batch)[:,-1:,:]
                # Sample inter-event time
                inter_time_dist = self.target_model.get_inter_time_dist(context)
                next_inter_time = inter_time_dist.sample()  # [batch_size, 1]
                
                # Sample mark if needed
                if current_batch.marks is not None:
                    mark_logits = torch.log_softmax(self.target_model.mark_linear(context), dim=-1)
                    mark_dist = torch.distributions.Categorical(logits=mark_logits)
                    next_mark = mark_dist.sample()  # [batch_size, 1]
                
                # Calculate new event time
                next_time = current_batch.event_times[:, -1:] + next_inter_time
                
                # Check if any new times exceed t_end
                if (next_time >= t_end).any():
                    # Truncate events past t_end
                    valid_indices = (next_time < t_end).squeeze(-1)
                    
                    if not valid_indices.any():
                        # No valid times left
                        break
                    
                    # Keep only valid items in batch
                    next_time = next_time[valid_indices]
                    next_inter_time = next_inter_time[valid_indices]
                    if current_batch.marks is not None:
                        next_mark = next_mark[valid_indices]
                
                # Update batch with new event
                new_event_times = torch.cat([current_batch.event_times, next_time], dim=1)
                new_mask = torch.cat([current_batch.mask, torch.ones_like(next_time)], dim=1)
                
                # Calculate inter-event times from event times
                new_inter_times = torch.ones_like(new_event_times)
                new_inter_times[:, 1:] = torch.diff(new_event_times, dim=1)

                if current_batch.marks is not None:
                    new_marks = torch.cat([current_batch.marks, next_mark], dim=1)
                    current_batch = Batch(
                        inter_times=new_inter_times,
                        event_times=new_event_times,
                        mask=new_mask,
                        marks=new_marks,
                        t_ends=current_batch.t_ends
                    )
                else:
                    current_batch = Batch(
                        inter_times=new_inter_times,
                        event_times=new_event_times,
                        mask=new_mask,
                        t_ends=current_batch.t_ends
                    )
                    
                num_generated += 1
                # if (num_generated % self.gamma) == 0:
                #     print(f'time for standard sampling {self.gamma} from target model autoregressively : {time.time() - start_time:.5f}')
                
                # Check if we've reached t_end for all processes
                if (next_time >= t_end).all():
                    break
    
        return current_batch
    def compute_distances(self, results):
        """
        Compute KS distance and Wasserstein distance between sampled and real sequences.
        
        Args:
            results: Dictionary containing sampling results
        
        Returns:
            Dictionary containing computed distances
        """
        def compute_empirical_cdf(times):
            # Sort times and compute empirical CDF
            sorted_times = np.sort(times)
            n = len(sorted_times)
            cumsum = np.arange(1, n + 1) / n
            return sorted_times, cumsum
        
        def compute_wasserstein_continuous(times1, times2):
            # Compute empirical CDFs
            x1, f1 = compute_empirical_cdf(times1)
            x2, f2 = compute_empirical_cdf(times2)
            
            # Combine all unique x values
            x = np.unique(np.concatenate([x1, x2]))
            
            # Interpolate CDFs to common x values
            f1_interp = np.interp(x, x1, f1)
            f2_interp = np.interp(x, x2, f2)
            
            # Compute Wasserstein distance
            dx = np.diff(x)
            areas = np.abs(f1_interp[:-1] - f2_interp[:-1]) * dx
            return np.sum(areas)
    
        def compute_wasserstein_discrete(marks1, marks2, num_marks):
            # Compute empirical probabilities for each mark type
            pi1 = np.bincount(marks1, minlength=num_marks) / len(marks1)
            pi2 = np.bincount(marks2, minlength=num_marks) / len(marks2)
            
            # Compute Wasserstein distance for discrete marks
            return 0.5 * np.sum(np.abs(pi1 - pi2))
        
        distances = {}
        
        # Extract times and marks
        spec_times = results['speculative']['times'].cpu().numpy().flatten()
        std_times = results['standard']['times'].cpu().numpy().flatten()
        real_times = results['real']['times'].cpu().numpy().flatten()

        # Compute Wasserstein distances for times
        distances['ws_time_spec'] = compute_wasserstein_continuous(spec_times, real_times)
        distances['ws_time_std'] = compute_wasserstein_continuous(std_times, real_times)
        
        # If marks are present, compute Wasserstein distances for marks
        if results['speculative']['marks'] is not None:
            spec_marks = results['speculative']['marks'].cpu().numpy().flatten()
            std_marks = results['standard']['marks'].cpu().numpy().flatten()
            real_marks = results['real']['marks'].cpu().numpy().flatten()
            
            num_marks = self.target_model.num_marks
            distances['ws_mark_spec'] = compute_wasserstein_discrete(spec_marks, real_marks, num_marks)
            distances['ws_mark_std'] = compute_wasserstein_discrete(std_marks, real_marks, num_marks)
        
        return distances

    def benchmark_comparison(self, init_batch, batch_size=1, t_end=100, max_events=1000):
        """
        Compare performance between speculative decoding and standard autoregressive sampling.
        
        Args:
            init_batch: Batch object containing initial historical event data
            batch_size: Number of independent point processes to simulate
            t_end: Upper time bound for the simulation
            max_events: Maximum number of events to generate
            
        Returns:
            results: Dictionary with benchmark results
        """
        # Run standard autoregressive sampling with target model
        print("Running standard autoregressive sampling...")
        std_start = time.time()
        std_batch = self.standard_sample(init_batch, batch_size, t_end, max_events)
        std_duration = time.time() - std_start
        print(f"Standard sampling took {std_duration:.2f} seconds")

        # Run speculative decoding
        print("Running speculative decoding sampling...")
        spec_start = time.time()
        spec_batch, spec_metrics = self.sample(init_batch, batch_size, t_end, max_events)
        spec_duration = time.time() - spec_start
        print(f"Speculative decoding took {spec_duration:.2f} seconds")
        
        # Calculate speedup
        speedup = std_duration / spec_duration
        
        # Calculate number of generated events
        initial_events = 1  # We start with one event from init_batch
        spec_events_generated = spec_batch.event_times.shape[1] - initial_events
        std_events_generated = std_batch.event_times.shape[1] - initial_events
        
        # Compute NLL for both methods
        batches = {'standard': std_batch, 'speculative': spec_batch}
        ll_results = self.compute_ll(batches)

        # Compile results
        results = {
            "speculative": {
                "times": spec_batch.event_times,
                "marks": spec_batch.marks,
                "duration": spec_duration,
                "metrics": spec_metrics,
                "num_events": spec_events_generated
            },
            "standard": {
                "times": std_batch.event_times,
                "marks": std_batch.marks,
                "duration": std_duration,
                "num_events": std_events_generated
            },
            "real": {
                "times": init_batch.event_times,
                "marks": init_batch.marks if init_batch.marks is not None else None
            },
            "speedup": speedup,
            "ll":{
                "speculative": ll_results["speculative"],
                "standard": ll_results["standard"]
            }
        }
        distances = self.compute_distances(results)
        results['distances'] = distances
        return results
    
    def compute_ll(self, batches):
        """
        Calculate the negative log-likelihood (NLL) of sampled point processes using the target model.
        Args:
            batches: Dictionary with sampling method keys and corresponding Batch objects
            
        Returns:
            Dictionary with NLL statistics for each sampling method
        """
        self.target_model.eval()
        results = {}
        with torch.no_grad():
            for method, batch in batches.items():
                log_probs = self.log_prob_verify(self.target_model, batch)
                ll = log_probs.mean().item()
                results[method] = ll
        
        return results

    def log_prob_verify(self, model:dpp.models.LogNormMixTransformer, batch: dpp.data.Batch):
        context = model.get_context(batch)  # (batch_size, seq_len, context_size)
        inter_time_dist = model.get_inter_time_dist(context) # A LogNormalMixtureDistribution object
        inter_times = batch.inter_times.clamp(1e-10)        # avoid log(0)
        log_p = inter_time_dist.log_prob(inter_times)

        # Compute the survival log-likelihood from t_N to t_end
        # A sample mask: [[1,1,1,0], [1,1,0,0]]，therefore mask.sum(-1, keepdim=True) is [[3],[2]], indicating the index of the last event in each sequence
        # last_event_idx = batch.mask.sum(-1, keepdim=True).long()-1  # (batch_size, 1)
        # log_surv_all = inter_time_dist.log_survival_function(inter_times)  # (batch_size, seq_len), compute the survival likelihood from t_1 to t_N
        # log_surv_last = torch.gather(log_surv_all, dim=-1, index=last_event_idx).squeeze(-1)  # (batch_size,), we need the survival likelihood at the last event only

        if model.num_marks > 1:  # compute the log-likelihood of marks
            mark_logits = torch.log_softmax(model.mark_linear(context), dim=-1)  # MLP layer for classification, (batch_size, seq_len, num_marks)
            mark_dist = Categorical(logits=mark_logits)
            log_p += mark_dist.log_prob(batch.marks)  # (batch_size, seq_len)

        log_p *= batch.mask  # (batch_size, seq_len)
        
        # return (log_p.sum(-1) + log_surv_last) / batch.t_ends  # (batch_size,)
        return log_p.sum(-1) / batch.t_ends  # (batch_size,)
    
    def rejection_sampling(self, target_dist, draft_dist=None, max_attempts=50):
        for attempt in range(max_attempts):
            proposal = target_dist.sample()
            
            # acceptance prob \alpha = max(0, p(x) - q(x)) / p(x)
            target_prob = target_dist.log_prob(proposal).exp()
            draft_prob = draft_dist.log_prob(proposal).exp()
            
            modified_prob = torch.clamp(target_prob - draft_prob, min=0.0)

            acceptance_ratio = torch.where(
                target_prob > 0,
                modified_prob / target_prob,
                torch.zeros_like(modified_prob)
            )

            u = torch.rand_like(acceptance_ratio)
            if u.item() <= acceptance_ratio.item():
                return proposal
        
        print("Warning: Reached maximum attempts for rejection sampling, returning sample from target distribution")
        return target_dist.sample()

    def stupid_sampling(self, target_dist, draft_dist=None):
        return target_dist.sample()

def main(config):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # Set up random seeds
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load dataset
    dataset_name = config['dataset_name']
    batch_size = config['sampling']['batch_size']
    t_end = config['sampling']['t_end']
    gamma = config['sampling']['gamma']
    
    # Load dataset
    dataset = dpp.data.load_dataset(dataset_name)
    d_train, d_val, d_test = dataset.train_val_test_split(seed=seed)
    
    # Get dataloader
    dl_test = d_test.get_dataloader(batch_size=1, shuffle=False)
    
    # Get dataset statistics for model initialization
    mean_log_inter_time, std_log_inter_time = d_test.get_inter_time_statistics()
    config['num_marks'] = d_test.num_marks
    config['mean_log_inter_time'] = mean_log_inter_time
    config['std_log_inter_time'] = std_log_inter_time
    
    # Load models
    target_model, draft_model = load_models(config)
    
    # Create sampler
    sampler = SpeculativeTPPSampler(target_model, draft_model, gamma=gamma)
    
    # Get a test batch for initialization
    if config['sampling']['init_batch']:
        init_batch = next(iter(dl_test))
    else:
        print("Sampling without history")
        init_batch = Batch(
            inter_times=torch.zeros(1, 1),
            event_times=torch.zeros(1, 1),
            mask=torch.ones(1, 1),
            t_ends=torch.full((1,), t_end)
        )

    # Run benchmark
    results = sampler.benchmark_comparison(
        init_batch,
        batch_size=batch_size,
        t_end=t_end,
        max_events=config['sampling']['max_events']
    )
    
    # Print results
    print(f"Standard sampling (AR) took: {results['standard']['duration']:.2f} seconds") # <-- Add this line
    print(f"Speculative decoding (SD) took: {results['speculative']['duration']:.2f} seconds") # <-- Optional: Clarify SD time as well
    print(f"Speculative decoding speedup: {results['speedup']:.2f}x")
    print(f"Acceptance rate: {results['speculative']['metrics']['acceptance_rate']:.2f}")
    print(f"Events per iteration: {results['speculative']['metrics']['events_per_iteration']:.2f}")
    
    add_synth_dataset(dl_test, results, config)

     # Print distance metrics
    print("\nDistance Metrics:")
    print(f"Time Wasserstein distance (Standard): {results['distances']['ws_time_std']:.4f}")
    print(f"Time Wasserstein distance (Speculative): {results['distances']['ws_time_spec']:.4f}")

    # Plot results
    output_path = Path(config['plotting']['output_dir']) / (dataset_name.split('/')[-1] + '_speculative')
    output_path.mkdir(parents=True, exist_ok=True)
    plot_path = output_path / f"speculative_tpp_gamma={gamma}.png"
    plot_sampling_results(results, save_path=plot_path)

    # log_qqplot_path = output_path / f"log_qq_plot_gamma={gamma}.png"
    # correlation = plot_log_qqplot(results, save_path=log_qqplot_path)
    # print(f"Distribution correlation (log scale): {correlation:.4f}")

    log_gof_path = output_path / f"log_gof_gamma={gamma}.pdf"
    gof_metrics = plot_temporal_goodness_of_fit(results, type=config['sampling']['type'], save_path=log_gof_path)
    print(f"KS-stat for standard sampling: {gof_metrics['ks_stat_std']:.4f}, KS-stat for SD: {gof_metrics['ks_stat_spec']:.4f}, KS-stat for real data: {gof_metrics['ks_stat_real']:.4f}")

    print("\nRelative Difference (D):")
    #gof_metrics = plot_temporal_goodness_of_fit(results, type=config['sampling']['type'], save_path=log_gof_path)
    dks = gof_metrics['ks_stat_spec'] - gof_metrics['ks_stat_std']
    dws = results['distances']['ws_time_spec'] - results['distances']['ws_time_std']
    d_ratio = dks / dws if dws != 0 else float('inf')
    print(f"D = {d_ratio:.4f}")

    nll_barchart = output_path / f"ll_barchart_gamma={gamma}.png"
    plot_ll_comparison(results, save_path=nll_barchart)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample from a trained point process model using speculative decoding')
    parser.add_argument('--config', type=str, default='scripts/sd_config_myhawkes.yaml', help='Configuration file path')
    # parser.add_argument('--config', type=str, default='scripts/sd_config_inhomo_poi.yaml', help='Configuration file path')
    # parser.add_argument('--config', type=str, default='scripts/sd_config_multi_hawkes.yaml', help='Configuration file path')
    # parser.add_argument('--config', type=str, default='scripts/sd_config_self_correct.yaml', help='Configuration file path')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    main(config)