# profanity: vanity account and contract addresses

Searches private-key offsets so that `seed_key + offset` controls an address matching your pattern.

## Why this is safe to run on someone else's GPU

The miner never accepts, generates or sees a private key. You create a keypair offline, hand over only the **public** key, and get back an **offset**. The private key exists only on your machine, both before and after.

This is the whole reason profanity2 exists: the original profanity generated private keys from a 32-bit seed, so every address it ever produced was brute-forceable, and funds were stolen. Passing a public key removes the possibility by construction.

## 1. Generate a keypair offline

```bash
openssl ecparam -genkey -name secp256k1 -text -noout -outform DER | xxd -p -c 1000 \
  | sed 's/41534e31204f49443a20736563703235366b310a//' \
  | sed 's/^30740201010420/PRIVATE: /' \
  | sed 's/a00706052b8104000aa144034200/\nPUBLIC: /'
```

Keep the private half somewhere safe and offline. The public half is 128 hex characters with no `04` prefix; that is what the miner wants.

If you already have a private key you want to use as the seed, `cast wallet public-key` derives its public key. Run it on a machine you trust, since the private key has to be present for it.

## 2. Mine

```bash
1miner profanity --public-key <128 hex chars> --leading 0
```

Useful flags:

- `--contract` scores the contract this key would deploy rather than the account address itself. Use it when you want a vanity *contract* address from a fresh deployer. See the warning below, because it carries a condition.
- `--exact <mask>` reports every address matching the mask in full instead of climbing towards a best score. This is often what you want here: rather than watching the score creep up, you state the pattern you will accept and collect matches as they arrive.
- `-i` / `--inverse-size` and `-I` / `--inverse-multiple` set the batch geometry. Their product is the number of points per round, and it dominates both memory use and start-up time. Defaults are 255 and 16384, about 4.2M points and roughly 400 MB of device memory. Lower `-I` first if you are short on memory.

Nothing stops on its own, so pass `--seconds` for a bounded run or interrupt it when you have what you wanted.

Output looks like:

```
  Time:     2s  Score:  8  GPU0  Offset: 0x0000778a...8fb4  Address: 0x000000007AA7965814DC341e6DD4Fb6213179936
```

### `--contract` assumes the account has never sent a transaction

A CREATE address is derived from the deploying account and its nonce, and `--contract` scores the address for **nonce 0**. So the contract only lands where the miner said if the deployment is that account's very first transaction.

Send anything else from it first — a test transfer, an approval, a failed attempt — and the nonce has moved on, so the mined address is gone for good. Fund the account, then deploy, and do nothing else with it in between. You can check what an account would deploy to at any nonce with:

```bash
cast compute-address <ADDRESS> --nonce 0
```

## 3. Turn the offset into a private key

Add the offset to your seed private key modulo the secp256k1 group order:

```
final_private_key = (seed_private_key + offset) mod n
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
```

Any bignum tool will do. In Python:

```python
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
final = (int(seed_priv, 16) + int(offset, 16)) % n
print(hex(final))
```

## 4. Verify before you fund it

Import the resulting private key into a wallet and confirm it shows the address the miner reported. `1miner` already re-derives every hit on the CPU before printing it, so a mismatch here would be surprising, but this is the step that protects you against a bug in software you did not write. Do it before sending anything of value.

## Notes

- The offset's top 16 bits are always zero, so adding it to a seed key cannot overflow 256 bits.
- Each GPU gets its own slice of the offset space rather than a random starting point of its own, so no two devices in a run can cover the same ground.
- Metal does not support this mode. It needs a secp256k1 kernel, and only the keccak-based salt modes have one. Use `--backend opencl`.
- `--backend cpu` works and is useful for verification, but it performs a modular inversion per step and is orders of magnitude slower.
- The kernel deliberately skips the equal-x edge cases in point addition, as the reference implementation does. Those cases are astronomically rare and cost only a missed candidate.
