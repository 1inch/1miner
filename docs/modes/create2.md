# create2: vanity CREATE2 addresses

Searches `bytes32` salts so that a known deployer, deploying known init code, lands on an address matching your pattern.

```
address = keccak256(0xff ++ deployer ++ salt ++ keccak256(initCode))[12:]
```

Because the init code is part of the pre-image, **the address changes if the contract changes**. Compile the exact bytecode you intend to deploy before you start mining; a constructor argument or a compiler version bump invalidates the salt. If that is inconvenient, use [create3](create3.md), where the address does not depend on the init code at all.

## Mining

You need the deployer and the init code, in any of three forms:

```bash
# Init code inline.
1miner create2 --deployer 0xYourFactory --init-code 0x60806040... --leading 0

# Init code from a file, which is easier for real contracts.
1miner create2 --deployer 0xYourFactory --init-code-file build/MyToken.hex --matching dead

# If you already have the hash.
1miner create2 --deployer 0xYourFactory --init-code-hash 0xYourInitCodeHash --zero-bytes
```

`--deployer` is required. It is never assumed, because a salt mined against the wrong deployer produces an address that looks entirely valid and is unusable.

Every scoring mode works here, including `--exact` to keep reporting every address that matches a mask in full rather than climbing towards a best score. See [backends.md](../backends.md) for the tuning flags and [benchmarking.md](../benchmarking.md) for what rate to expect. CREATE2 hashes once per candidate, so it is the fastest of the four modes.

Getting the init code hash from Foundry:

```bash
cast keccak "$(forge inspect MyToken bytecode)"
```

Remember that init code includes constructor arguments. With arguments, build the full creation payload rather than the bare bytecode.

## Using the result

The miner prints a salt:

```
  Time:    31s  Score: 8  GPU0  Salt: 0x8954...2441  Address: 0x00000000834E17ea2F65c7ddc423DceAFa664a76
```

Check the prediction independently before deploying. `cast` computes the same thing from the same inputs, so agreement means your deployer and init code were what you thought:

```bash
cast create2 --deployer 0xYourFactory --salt 0x8954...2441 --init-code 0x60806040...
```

Then deploy through whatever CREATE2 entry point your factory exposes, for example:

```bash
cast send 0xYourFactory "deploy(bytes32,bytes)" 0x8954...2441 0x60806040... \
  --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY"
```

Deployment fails if something already occupies the address, which is worth knowing if you mined a short pattern that somebody else may have reached first.

## Three ways to waste a run

| Mistake | What you see | What it costs |
|---|---|---|
| Wrong `--deployer` | Nothing. Normal-looking addresses and a climbing score. | The whole run. |
| Init code that later changes | Nothing during mining; the deploy simply lands elsewhere. | The whole run. |
| Bare runtime bytecode instead of creation code, or missing constructor arguments | Nothing. It is a valid hash of the wrong thing. | The whole run. |

All three are silent, because all three produce perfectly well-formed addresses of a different deployment. The `cast create2` call above catches every one of them for the price of one command, so make it before you deploy rather than after.

## Front-running

A CREATE2 salt is only worth something to the account that can use it. If your factory lets anyone deploy with any salt, someone watching the mempool can take the address you mined. Factories usually guard this with an owner check or by folding `msg.sender` into the salt. Know which one yours does before you mine something valuable.
