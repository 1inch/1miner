# 1nft: 1inch Address NFT

Finds the input that makes the [1inch Address NFT](https://etherscan.io/address/0x1ADD4E55ecEffd795B01d22203D280c93A2F1dc3) deployer produce a vanity contract address for a particular account.

The deployer hands out vanity addresses as tokens: you search for an input that lands on an address you like, mint it, and the token records that the address is yours to deploy at. What you mine here is that input.

## How the scheme works

The deployer is a CREATE3 factory, so an address depends on exactly two things: the deployer contract and a 32-byte salt. The init code of whatever you eventually deploy plays no part, which is why the address can be settled before that contract is even written. See [how-address-derivation-works.md](../how-address-derivation-works.md) for the derivation itself.

What makes this mode different from plain [create3](create3.md) is that the deployer does not let you choose the whole salt. It builds the salt from two halves, in [AddressToken.sol](https://github.com/1inch/address-token/blob/master/contracts/AddressToken.sol):

```solidity
function getTokenIdAndSalt(bytes16 magic, address account) public view returns(address tokenId, bytes32 salt) {
    bytes32 hashedAccount = keccak256(abi.encodePacked(account));
    salt = (_LOW_128_BIT_MASK & hashedAccount) | bytes32(magic);
    tokenId = CREATE3.getDeployed(salt);
}
```

`bytes32(magic)` is left-aligned, so the magic lands in the high 16 bytes and the masked account hash in the low 16:

```
        16 bytes                    16 bytes
┌───────────────────────┬───────────────────────────────┐
│  magic                │  keccak256(account)[16:32]    │
│  yours to choose      │  fixed by the account         │
└───────────────────────┴───────────────────────────────┘
                        salt
address = CREATE3(deployer, salt)
```

Only the high half is searchable. The low half is computed once from the account before mining starts and never changes during the run, so the search space is 16 bytes rather than 32. That is still far more than any pattern needs.

Note that the token id **is** the address: `tokenId = CREATE3.getDeployed(salt)`, and the contract stores the salt against it so the owner can deploy there later.

This is why the miner reports a **magic** rather than a salt: the magic is the only part you chose, and it is what `mint` takes.

## Why `--deployer` is required

The address is a function of the deployer contract. Point the search at a different deployer and every address it finds is different, so a magic mined against the wrong one mints nothing.

Nothing about that failure is visible in the output. The addresses look right, the score climbs, the run finishes, and the magic is worthless. 1miner therefore never assumes a deployer, not even the well-known 1inch one, because the earlier tool that did default to it made exactly this mistake easy to make. For the live 1inch Address NFT contract the value is `0x1ADD4E55ecEffd795B01d22203D280c93A2F1dc3`; check it on the linked page rather than copying it from memory.

## What `--mint-for` is

It is the account the address is minted for: the one that receives the token, and so the one that may later deploy at the address.

Whoever sends the mint transaction is irrelevant. The address belongs to the account you name here, so a friend, a relayer or a paymaster can pay the gas without gaining any claim to it.

The binding exists as front-running protection. Your magic becomes public the moment the mint transaction is broadcast. If the whole salt were free to choose, anyone watching could take that magic and mint the same address first. Deriving half of the salt from the account removes the incentive: a magic only ever produces that address for that one account, so publishing it costs you nothing.

The consequence is worth stating plainly: **a magic belongs to one account.** Name a different account and the address changes completely, so mining for the wrong one means starting over. Once minted, though, the token is ordinary and transferable — so minting to another account of your own is recoverable by transferring it afterwards, while minting to an account you do not control is not.

## Mining

```bash
1miner 1nft \
  --deployer 0x1ADD4E55ecEffd795B01d22203D280c93A2F1dc3 \
  --mint-for 0xYourAccount \
  --leading 0
```

Both flags are required. The only exception is `--benchmark`, where the output is a throughput figure rather than a usable magic, so neither has to be supplied.

Every scoring mode works here: `--leading`, `--matching`, `--zero-bytes`, `--mirror` and the rest. See [backends.md](../backends.md) for the tuning flags and [benchmarking.md](../benchmarking.md) for what rate to expect. Being CREATE3, this mode hashes twice per candidate and so runs at roughly half the rate of create2.

## The result

```
  Time:     0s  Score:  5  GPU0  Magic: 0xd0d3cef9872bfa6d4a1b7cf826b68d9f  Address: 0x44179788579a5D52006eFA9500Af423400700000
```

That line is a real hit for the 1inch deployer and account `0x00000000219ab540356cbb839cbe05303d7705fa`, scored on zero bytes. Every hit is re-derived on the CPU before it is printed, so a magic that does not actually produce the address shown is reported as an error rather than as a result. The magic is checked against `--mint-for` too, by rebuilding the salt the way the deployer does, which is what a hit reported by a device you do not trust could otherwise get wrong; that check runs even under `--no-verify`.

## Check it against the contract before you mint

The contract will tell you what your magic produces, for free. `getTokenIdAndSalt` is a `view` function, so this costs nothing and settles both flags at once:

```bash
cast call 0x1ADD4E55ecEffd795B01d22203D280c93A2F1dc3 \
  "getTokenIdAndSalt(bytes16,address)(address,bytes32)" \
  0xd0d3...8d9f 0xYourAccount --rpc-url "$RPC_URL"
```

If the address it returns is the one the miner printed, both `--deployer` and `--mint-for` were right. If it is not, one of them was wrong and the magic is useless. This is the check that catches the two silent mistakes below, and it is worth one call before you spend gas.

## Minting it

Which function you call depends only on who sends the transaction:

```bash
# Sent from the account you mined for.
cast send 0x1ADD4E55ecEffd795B01d22203D280c93A2F1dc3 "mint(bytes16)" 0xd0d3...8d9f \
  --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY"

# Sent from any account, on behalf of the one you mined for.
cast send 0x1ADD4E55ecEffd795B01d22203D280c93A2F1dc3 "mintFor(bytes16,address)" \
  0xd0d3...8d9f 0xYourAccount \
  --rpc-url "$RPC_URL" --private-key "$PRIVATE_KEY"
```

Use `mint` only when the sending account is the one in `--mint-for`, since it passes the sender as the account. `mintFor` states the account explicitly and is the safer choice.

An address can be minted only once: a second attempt reverts with `RemintForbidden`. So a pattern short enough to be found quickly may already be taken, and the `getTokenIdAndSalt` call above plus a look at the token on a marketplace tells you before you try.

## Deploying at the address

Only the token owner can deploy, and doing so burns the token:

```solidity
function deploy(address tokenId, bytes calldata creationCode) public payable returns(address deployed);
function deployAndCalls(address tokenId, bytes calldata creationCode, bytes[] calldata cds) external payable;
```

The token id is the address itself, so pass the mined address as `tokenId`. `deployAndCalls` deploys and then makes a series of calls to the new contract in the same transaction, which is convenient for initialising it. Since the address comes from CREATE3, `creationCode` plays no part in where it lands — you can decide what to deploy long after minting.

Transferring the token does not affect the address. The salt is written to `salts[tokenId]` at mint time and `deploy` reads it back from there, so it is never recomputed from whoever is deploying; and the CREATE2 deployer inside CREATE3 is the AddressToken contract, not the caller. So `A.mint(magic)`, `transferFrom(A, B, tokenId)`, then `B.deploy(tokenId, code)` lands the contract at exactly the minted address. Note that the ERC721 token id is the address widened to `uint256`, and that `deploy` burns the token, so it works once.

## Two ways to waste a run

| Mistake | What you see | What it costs |
|---|---|---|
| Wrong `--deployer` | Nothing. Normal-looking addresses, a climbing score, a magic that mints nothing. | The whole run. |
| Wrong `--mint-for` | Nothing, for the same reason. | The whole run. |

Both are silent because both produce perfectly well-formed addresses; they are simply addresses of a different scheme. The `getTokenIdAndSalt` call above is the only thing that catches either, so make it before you mint rather than after.

## When you want plain create3 instead

Use [create3](create3.md) if you control a CREATE3 factory of your own and want the full 32-byte salt, with no account binding and nothing to mint. That path gives a salt you pass straight to `deploy(salt, initCode)`. Use `1nft` when the address is to come from the 1inch Address NFT deployer, be owned as a transferable token, and be tradeable on the secondary market the collection has.

The contract source is at [1inch/address-token](https://github.com/1inch/address-token/blob/master/contracts/AddressToken.sol) if you want to read the rest of it.
