# create3: vanity CREATE3 addresses

Searches `bytes32` salts for a CREATE3 factory. The address depends only on the factory and the salt:

```
proxy   = keccak256(0xff ++ factory ++ salt ++ keccak256(proxyBytecode))[12:]
address = keccak256(0xd6 0x94 ++ proxy ++ 0x01)[12:]
```

The factory deploys a minimal proxy with CREATE2, and the proxy deploys your contract with CREATE at nonce 1. Since your init code never enters the calculation, **you can mine the address before the contract is written**, change the implementation freely, and deploy the same address on every chain where the factory exists at the same address.

The cost is throughput: two keccaks per candidate instead of one, so roughly half the rate of [create2](create2.md).

## Mining

```bash
1miner create3 --deployer 0xYourFactory --leading 0
```

`--deployer` is the factory and is required. It is never assumed, because a salt mined against the wrong factory produces an address that looks entirely valid and is unusable.

If you do not have a factory yet, the reference [`Create3Deployer`](https://github.com/1inch/create3-contract/blob/main/contracts/Create3Deployer.sol) comes with a deploy script. It exposes exactly two functions:

```solidity
function deploy(bytes32 salt, bytes calldata code) external onlyOwner returns (address);
function addressOf(bytes32 salt) external view returns (address);
```

`--bytecode-hash` defaults to `0x21c35dbe1b344a2488cf3321d6ce542f8e9f305544ff09e4993a62319a497c1f`, the hash of the standard Solady/solmate proxy `0x67363d3d37363d34f03d5260086018f3`. Almost every CREATE3 factory uses it. Override only if yours deploys a different proxy, and confirm first:

```bash
cast keccak 0x67363d3d37363d34f03d5260086018f3
```

A wrong proxy hash silently yields addresses that will never exist, so this is worth one command to check.

Every scoring mode works here, including `--exact` to report every address matching a mask in full rather than climbing towards a best score, repeatable to search several masks at once. See [backends.md](../backends.md) for the tuning flags and [benchmarking.md](../benchmarking.md) for what rate to expect.

## Checking your setup against a known answer

If you want to confirm the derivation before committing to a long run, this vector is asserted in this repository's tests and cross-checked against both the Solidity factory and `cast`:

```bash
# factory 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf with a zero salt
# gives 0x6c8Ed9dC3734d7944BEDDd2fB5AcdF5f17247870

# the intermediate CREATE2 proxy, which cast can compute independently
cast create2 --deployer 0x9fBB3DF7C40Da2e5A0dE984fFE2CCB7C47cd0ABf \
  --salt 0x0000000000000000000000000000000000000000000000000000000000000000 \
  --init-code 0x67363d3d37363d34f03d5260086018f3
# 0x932A2198eC22043b9702a6250C8Ad906a3D62131

# then the CREATE step at nonce 1 from that proxy
cast compute-address 0x932A2198eC22043b9702a6250C8Ad906a3D62131 --nonce 1
# 0x6c8Ed9dC3734d7944BEDDd2fB5AcdF5f17247870
```

That factory address is a test fixture rather than a deployed contract, so do not mine against it expecting to use the result. `1miner self-test` checks the same vector for you.

## Using the result

```
  Time:     6s  Score:  8  GPU0  Salt: 0x8954...2441  Address: 0x00000000834E17ea2F65c7ddc423DceAFa664a76
```

Confirm against the factory itself, which is the authoritative answer:

```bash
cast call 0xFactory "addressOf(bytes32)(address)" 0x8954...2441 --rpc-url "$RPC_URL"
```

Then deploy:

```bash
cast send 0xFactory "deploy(bytes32,bytes)" 0x8954...2441 0x60806040... \
  --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY"
```

## Same address across chains

The address is a function of the factory address and the salt, so the same salt gives the same address on every chain where the factory sits at the same address. That is the usual reason to pick CREATE3 over CREATE2. It only holds while the factory address matches, so deploy the factory itself deterministically if you care about this.

## Two ways to waste a run

| Mistake | What you see | What it costs |
|---|---|---|
| Wrong `--deployer` | Nothing. Normal-looking addresses and a climbing score, for a factory that will never produce them. | The whole run. |
| Wrong `--bytecode-hash` | Nothing, for the same reason. | The whole run. |

Both are silent because both produce perfectly well-formed addresses; they are simply addresses of a different factory. Confirm the pair with `addressOf` on your own factory before mining anything valuable, and the known-answer vector above if you want to check the derivation itself.

## Front-running

Since the address does not depend on init code, anyone who can call your factory with your salt can take the address and put arbitrary code at it. This is sharper than the equivalent CREATE2 risk, where at least the init code has to match. The reference `Create3Deployer` restricts `deploy` to the owner, which is the usual answer. Check what yours does before mining something valuable.
