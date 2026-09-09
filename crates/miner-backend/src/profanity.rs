//! What every profanity backend needs and none of them should write twice.
//!
//! Each work item starts at `seed_pub + (seed + (id << 192)) * G` and every
//! round advances all points by one generator step, so the scalar for a hit is
//! `seed + round + (id << 192)`. That offset is what gets reported: added to
//! the user's seed private key it yields the private key for the found address,
//! and the miner never sees a private key at any point.
//!
//! Devices are partitioned inside that offset rather than left to chance: the
//! top lane of `seed` carries a device slot above the bits the kernel adds `id`
//! into, so two devices cannot walk the same sequence however they are drawn.
//!
//! Both the offset arithmetic and the result-slot layout live here rather than
//! in one backend, because a backend that gets either subtly wrong hands the
//! user a key controlling a different address, and the two implementations
//! would fail differently. The GPU-side field arithmetic is deliberately not
//! shared: that is the part the cross-backend agreement tests exist to check.

use miner_core::{ProfanityConfig, ScoreSpec, secp256k1::generator_table};
use rand::RngCore;

use crate::{BackendError, DeviceInfo, EXACT_CAPACITY, Hit, Job, MAX_SCORE, Progress, Result};

/// `mp_number` from profanity2's types.hpp: eight 32-bit words, least
/// significant first, 16-byte aligned. Mirrors `MpNumber` in both kernels.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct MpNumber {
    pub d: [u32; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MpPoint {
    pub x: MpNumber,
    pub y: MpNumber,
}

/// One result slot, matching `result` in profanity.cl and `ProfResult` in
/// profanity.metal.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ResultSlot {
    pub found: u32,
    pub found_id: u32,
    pub found_hash: [u8; 20],
}

/// Four 64-bit lanes, least significant first. OpenCL takes this as a `ulong4`
/// kernel argument; Metal reads it as a field of its params struct.
#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
pub struct Ulong4(pub [u64; 4]);

/// Big-endian 32 bytes into four 64-bit lanes, least significant lane first.
pub fn be_bytes_to_ulong4(bytes: &[u8; 32]) -> Ulong4 {
    let mut lanes = [0u64; 4];
    for (i, lane) in lanes.iter_mut().enumerate() {
        let start = 24 - i * 8;
        *lane = u64::from_be_bytes(bytes[start..start + 8].try_into().unwrap());
    }
    Ulong4(lanes)
}

/// Big-endian 32 bytes into the eight little-endian words a kernel reads.
pub fn to_mp(be: &[u8; 32]) -> MpNumber {
    let mut d = [0u32; 8];
    for (i, word) in d.iter_mut().enumerate() {
        let start = 28 - i * 4;
        *word = u32::from_be_bytes(be[start..start + 4].try_into().unwrap());
    }
    MpNumber { d }
}

/// The generator table both backends upload, in the kernels' own layout.
pub fn precomp_table() -> Vec<MpPoint> {
    generator_table()
        .iter()
        .map(|p| {
            let (x, y) = p.to_bytes();
            MpPoint {
                x: to_mp(&x),
                y: to_mp(&y),
            }
        })
        .collect()
}

/// Enqueues the seeding pass is split into.
///
/// Seeding walks the precomp table with a full modular inversion per point
/// added, orders of magnitude more work per item than a round. Left as one
/// launch it can run long enough for a GPU watchdog to take it for a hang.
const INIT_CHUNKS: usize = 20;

/// How wide one seeding enqueue may be, given the round size and `--work-max`.
pub fn init_chunk(size: usize, work_max: usize) -> usize {
    (size / INIT_CHUNKS).clamp(1, work_max)
}

/// The most significant lane of an offset is a packed field. From the top: 16
/// bits left clear so `seed_priv + offset` cannot overflow 256 bits, 16 bits of
/// device slot, and 32 bits the kernel adds the work-item id into.
const ID_BITS: u32 = 32;
const MAX_ROUND_SIZE: u64 = 1 << ID_BITS;
const MAX_DEVICES: u64 = 1 << 16;

/// One device's starting offset: 192 random bits, so two runs do not cover the
/// same ground, above a device slot no other device of this run can reach.
///
/// Cryptographic quality is not needed — the security of the result comes from
/// the user's own seed key, which never enters this process — but `rand::rng()`
/// is per-thread and OS-seeded, unlike a clock read, and it is what the salt
/// modes already use.
pub fn device_seed(device_index: usize) -> Ulong4 {
    let mut bytes = [0u8; 32];
    rand::rng().fill_bytes(&mut bytes);
    let mut lanes = be_bytes_to_ulong4(&bytes);
    lanes.0[3] = (device_index as u64) << ID_BITS;
    lanes
}

/// Both fields have to hold for the separation to mean anything: a round wider
/// than its id field would reach into the next device's slot, and a slot above
/// its own field into the bits that must stay clear. Neither limit is anywhere
/// near a tuning that fits in memory — the default round is 2²² work items —
/// but checking them is what makes the separation structural rather than
/// assumed.
pub fn check_offset_fields(round_size: usize, infos: &[DeviceInfo]) -> Result<()> {
    if round_size as u64 > MAX_ROUND_SIZE {
        return Err(BackendError::Other(format!(
            "--inverse-size x --inverse-multiple is {round_size} work items, \
             above the {MAX_ROUND_SIZE} one round can address"
        )));
    }
    let highest = infos.iter().map(|i| i.index).max().unwrap_or(0) as u64;
    if highest >= MAX_DEVICES {
        return Err(BackendError::Other(format!(
            "device index {highest} is above the {MAX_DEVICES} an offset can keep apart"
        )));
    }
    Ok(())
}

/// `seed + round + (found_id << 192)` as a big-endian 32-byte scalar.
///
/// profanity2 open-codes this with a shortcut carry that misfires when a lane
/// is already zero; a full 256-bit add is used here instead.
pub fn offset_scalar(seed: &Ulong4, round: u64, found_id: u32) -> [u8; 32] {
    let mut lanes = seed.0;
    let mut carry = round as u128;
    for lane in lanes.iter_mut() {
        let sum = *lane as u128 + (carry & 0xFFFF_FFFF_FFFF_FFFF);
        *lane = sum as u64;
        carry = (carry >> 64) + (sum >> 64);
    }
    lanes[3] = lanes[3].wrapping_add(found_id as u64);

    let mut out = [0u8; 32];
    for (i, lane) in lanes.iter().enumerate() {
        let start = 24 - i * 8;
        out[start..start + 8].copy_from_slice(&lane.to_be_bytes());
    }
    out
}

/// What turning a result slot into a `Hit` needs beyond the slot itself.
pub struct RoundContext<'a> {
    pub cfg: &'a ProfanityConfig,
    pub job: &'a Job,
    pub device_index: usize,
    pub seed: &'a Ulong4,
    pub round: u64,
}

impl RoundContext<'_> {
    /// Build the hit a filled result slot describes.
    ///
    /// The offset is rebuilt from the seed, the round and the work-item id
    /// rather than read off the kernel, so walking the seed public key forward
    /// by it and comparing is what catches offset accounting that has gone
    /// wrong — which would otherwise hand over a key controlling a different
    /// address.
    fn hit_from(&self, slot: &ResultSlot, score: u32, pattern: Option<usize>) -> Hit {
        let mut address = [0u8; 20];
        address.copy_from_slice(&slot.found_hash);
        let offset = offset_scalar(self.seed, self.round, slot.found_id);

        Hit {
            score,
            address,
            salt: None,
            magic: None,
            offset: Some(offset),
            pattern,
            device_index: self.device_index,
            verified: !self.job.verify || self.cfg.address_for_offset(&offset) == Some(address),
        }
    }

    /// Result slots are indexed by score, so the best hit is the highest
    /// occupied slot above what has already been reported.
    ///
    /// The slot index is the device's own account of the score, and it is
    /// checked here against the host's scorer for the reason
    /// [`crate::salt::SaltRound::take_best`] gives at length: the value is
    /// published into a bar every device reads back, so an inflated one costs
    /// the rest of the search. The `u32` handed back is the bar to adopt, which
    /// for a hit that failed a check is the threshold unchanged.
    pub fn take_best(&self, results: &[ResultSlot], threshold: u64) -> Option<(u32, Hit)> {
        for score in (1..=MAX_SCORE).rev() {
            if results[score].found == 0 {
                continue;
            }
            if score as u64 <= threshold {
                break;
            }
            let mut hit = self.hit_from(&results[score], score as u32, None);
            hit.verified &= miner_core::score(&self.job.score, &hit.address) == score as u32;
            let bar = if hit.verified {
                score as u32
            } else {
                threshold as u32
            };
            return Some((bar, hit));
        }
        None
    }

    /// Slots are the round's matches in arrival order; `total` is how many
    /// there were, including any the buffer had no room for. It is a separate
    /// argument because OpenCL counts in slot 0 of the result buffer and Metal
    /// in slot 0 of its atomic flag buffer.
    pub fn drain_exact(&self, results: &[ResultSlot], total: u32) -> Vec<Progress> {
        let stored = (total as usize).min(EXACT_CAPACITY);
        let masks = self.job.exact.as_deref().unwrap_or_default();

        let mut found: Vec<Progress> = (1..=stored)
            .map(|slot| {
                // `found` names the mask that matched in this layout, one-based
                // so an untouched slot is distinguishable from mask 0.
                let pattern = results[slot].found.saturating_sub(1) as usize;
                let score = masks.get(pattern).map_or(0, ScoreSpec::constrained_bytes);
                Progress::Hit(self.hit_from(&results[slot], score, Some(pattern)))
            })
            .collect();

        if let Some(dropped) = total.checked_sub(stored as u32).filter(|d| *d > 0) {
            found.push(Progress::Dropped {
                count: dropped,
                device_index: self.device_index,
            });
        }
        found
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn info(index: usize) -> DeviceInfo {
        DeviceInfo {
            index,
            name: String::new(),
            compute_units: 0,
            global_memory: 0,
            driver: None,
        }
    }

    #[test]
    fn offset_is_seed_plus_round_plus_shifted_id() {
        let seed = Ulong4([5, 0, 0, 0]);
        let offset = offset_scalar(&seed, 7, 0);
        assert_eq!(u64::from_be_bytes(offset[24..].try_into().unwrap()), 12);

        // found_id lands in the most significant lane, i.e. shifted by 192 bits.
        let with_id = offset_scalar(&seed, 0, 3);
        assert_eq!(u64::from_be_bytes(with_id[..8].try_into().unwrap()), 3);
        assert_eq!(u64::from_be_bytes(with_id[24..].try_into().unwrap()), 5);
    }

    /// profanity2's shortcut carry treats an already-zero lane as a carry out.
    /// A full add must not.
    #[test]
    fn carry_only_propagates_on_real_overflow() {
        let seed = Ulong4([u64::MAX, 0, 0, 0]);
        let offset = offset_scalar(&seed, 1, 0);
        assert_eq!(u64::from_be_bytes(offset[24..].try_into().unwrap()), 0);
        assert_eq!(u64::from_be_bytes(offset[16..24].try_into().unwrap()), 1);

        let no_carry = Ulong4([1, 0, 0, 0]);
        let offset = offset_scalar(&no_carry, 1, 0);
        assert_eq!(u64::from_be_bytes(offset[16..24].try_into().unwrap()), 0);
    }

    #[test]
    fn seed_clears_the_top_bits_so_a_sum_cannot_overflow() {
        for device in [0, 1, 7, MAX_DEVICES as usize - 1] {
            assert_eq!(device_seed(device).0[3] >> 48, 0);
        }
    }

    #[test]
    fn seed_reserves_the_top_lane_for_the_device_slot() {
        for device in [0, 1, 7, MAX_DEVICES as usize - 1] {
            assert_eq!(device_seed(device).0[3], (device as u64) << ID_BITS);
        }
    }

    /// The whole point of the packing: the widest permitted round on one device
    /// stops short of the next device's slot, so no work item of one device can
    /// land on an offset another device reaches.
    #[test]
    fn the_widest_round_stops_short_of_the_next_device_slot() {
        // The largest id a permitted round produces, as the kernel reports it.
        let widest = u32::try_from(MAX_ROUND_SIZE - 1).expect("a round must fit the uint foundId");
        // Identical low lanes, as if the RNG had failed both devices, and the
        // highest round against the lowest: only the top lane can separate them.
        let shared = |device: u64| Ulong4([9, 9, 9, device << ID_BITS]);

        for device in 0..4 {
            let last = offset_scalar(&shared(device), u64::MAX >> 1, widest);
            let first_of_next = offset_scalar(&shared(device + 1), 0, 0);
            assert!(
                last < first_of_next,
                "device {device} reaches into the next slot"
            );
        }
    }

    /// The predecessor derived all 256 bits from a clock read, so two device
    /// threads starting together usually drew the same seed.
    #[test]
    fn the_random_part_of_a_seed_differs_every_draw() {
        let drawn: HashSet<[u64; 4]> = (0..1000).map(|_| device_seed(0).0).collect();
        assert_eq!(drawn.len(), 1000);
    }

    #[test]
    fn a_round_or_a_rig_too_large_for_the_offset_fields_is_rejected() {
        assert!(check_offset_fields(MAX_ROUND_SIZE as usize, &[info(0)]).is_ok());
        assert!(check_offset_fields(MAX_ROUND_SIZE as usize + 1, &[info(0)]).is_err());

        let highest = MAX_DEVICES as usize - 1;
        assert!(check_offset_fields(1, &[info(0), info(highest)]).is_ok());
        assert!(check_offset_fields(1, &[info(0), info(highest + 1)]).is_err());
    }

    #[test]
    fn public_key_lanes_are_least_significant_first() {
        let mut be = [0u8; 32];
        be[31] = 1;
        assert_eq!(be_bytes_to_ulong4(&be).0, [1, 0, 0, 0]);
        let mut be = [0u8; 32];
        be[0] = 1;
        assert_eq!(be_bytes_to_ulong4(&be).0, [0, 0, 0, 1 << 56]);
    }

    /// Both kernels declare these layouts, and a mismatch would be read as
    /// garbage field elements rather than as an error.
    #[test]
    fn the_wire_layouts_are_what_the_kernels_declare() {
        assert_eq!(size_of::<MpNumber>(), 32);
        assert_eq!(size_of::<MpPoint>(), 64);
        assert_eq!(size_of::<ResultSlot>(), 28);
        assert_eq!(align_of::<MpNumber>(), 16);
    }

    /// The generator as the seed public key: its private half is 1, so a
    /// failing case here can be reproduced by hand.
    fn profanity_config() -> ProfanityConfig {
        ProfanityConfig {
            seed_public_key: miner_core::secp256k1::generator(),
            contract: false,
        }
    }

    fn profanity_job(score: ScoreSpec) -> Job {
        Job {
            mode: miner_core::ModeConfig::Profanity(profanity_config()),
            score,
            keccak: crate::KeccakVariant::Tuned,
            tuning: crate::Tuning::default(),
            duration: None,
            verify: true,
            exact: None,
        }
    }

    /// The same defect [`crate::salt`] carries a test for, in the mode the
    /// report did not mention: a slot is indexed by score here too, so a device
    /// can walk to a real address, name the offset that really reaches it, and
    /// still choose what the run believes that address is worth.
    #[test]
    fn a_score_the_address_did_not_earn_is_rejected() {
        let cfg = profanity_config();
        let seed = Ulong4([5, 0, 0, 0]);
        let (round, found_id) = (7u64, 3u32);

        // A slot an honest kernel would write: the address the reported offset
        // genuinely reaches, so the re-derivation has something true to agree
        // with and only the slot index is in question.
        let offset = offset_scalar(&seed, round, found_id);
        let address = cfg
            .address_for_offset(&offset)
            .expect("the walk reaches an address");
        let slot = ResultSlot {
            found: 1,
            found_id,
            found_hash: address,
        };

        let job = profanity_job(ScoreSpec::matching(&hex::encode(&address[..3])).unwrap());
        let context = || RoundContext {
            cfg: &cfg,
            job: &job,
            device_index: 0,
            seed: &seed,
            round,
        };

        let mut earned = vec![ResultSlot::default(); crate::RESULT_SLOTS];
        earned[3] = slot;
        let (bar, hit) = context().take_best(&earned, 0).expect("an occupied slot");
        assert!(hit.verified);
        assert_eq!((bar, hit.score), (3, 3));

        let mut planted = vec![ResultSlot::default(); crate::RESULT_SLOTS];
        planted[MAX_SCORE] = slot;
        let (bar, hit) = context().take_best(&planted, 0).expect("an occupied slot");
        assert!(!hit.verified);
        assert_eq!(bar, 0);

        // The offset really does name the address printed beside it, so the
        // re-derivation alone would have passed this.
        assert_eq!(
            cfg.address_for_offset(&hit.offset.unwrap()),
            Some(hit.address)
        );
    }

    /// The kernels index the table as `[byte_index * 255 + (value - 1)]`, so
    /// the first entry has to be the generator itself.
    #[test]
    fn the_precomp_table_starts_at_the_generator() {
        let table = precomp_table();
        assert_eq!(table.len(), 8160);
        let (x, _) = miner_core::secp256k1::generator().to_bytes();
        assert_eq!(table[0].x.d, to_mp(&x).d);
    }
}
