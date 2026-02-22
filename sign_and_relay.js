const { ethers } = require('ethers');

const FACTORY   = "0x0692eC85325472Db274082165620829930f2c1F9";
const FORWARDER = "0x6c7726e505f2365847067b17a10C308322Db047a";
const RPC       = "https://mainnet.base.org";

const DOMAIN = {
  name: "ArgueDotFunForwarder", version: "1",
  chainId: 8453, verifyingContract: FORWARDER,
};
const FORWARD_TYPES = {
  ForwardRequest: [
    { name: "from",     type: "address" },
    { name: "to",       type: "address" },
    { name: "value",    type: "uint256" },
    { name: "gas",      type: "uint256" },
    { name: "nonce",    type: "uint256" },
    { name: "deadline", type: "uint48"  },
    { name: "data",     type: "bytes"   },
  ],
};

async function postRelay(payload) {
  const https = require('https');
  const body  = JSON.stringify(payload);
  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: "api.argue.fun", path: "/v1/relay", method: "POST",
      headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
      lookup: (hostname, options, cb) => cb(null, [{address:"216.150.16.193",family:4}]),
    }, res => {
      let data = "";
      res.on("data", d => data += d);
      res.on("end", () => resolve(JSON.parse(data)));
    });
    req.on("error", reject);
    req.write(body); req.end();
  });
}

async function signAndRelay({ debateAddress, side, lockedAmount, unlockedAmount, argument, gasLimit = 800000 }) {
  const privkey  = require('fs').readFileSync(process.env.HOME + '/.arguedotfun/.privkey', 'utf8').trim();
  const wallet   = new ethers.Wallet(privkey);
  const address  = wallet.address;
  const provider = new ethers.JsonRpcProvider(RPC);

  const forwarder = new ethers.Contract(FORWARDER, ["function nonces(address) view returns (uint256)"], provider);
  const nonce     = await forwarder.nonces(address);

  const iface    = new ethers.Interface(["function placeBet(address,bool,uint256,uint256,string)"]);
  const calldata = iface.encodeFunctionData("placeBet", [
    debateAddress, side === "A",
    BigInt(lockedAmount || 0), BigInt(unlockedAmount || 0), argument,
  ]);

  const deadline = Math.floor(Date.now() / 1000) + 3600;
  const message  = {
    from: address, to: FACTORY, value: 0n,
    gas: BigInt(gasLimit), nonce, deadline, data: calldata,
  };

  const signature = await wallet.signTypedData(DOMAIN, FORWARD_TYPES, message);

  const payload = {
    request: {
      from: address, to: FACTORY, value: "0",
      gas: gasLimit.toString(), nonce: nonce.toString(),
      deadline: deadline.toString(), data: calldata,
    },
    signature,
  };

  const result = await postRelay(payload);
  return { address, nonce: nonce.toString(), deadline, sig: signature.slice(0,20)+"...", relay: result };
}

const [,, debate, side, locked, unlocked, ...argParts] = process.argv;
signAndRelay({
  debateAddress: debate, side: side || "A",
  lockedAmount: locked || "0", unlockedAmount: unlocked || "0",
  argument: argParts.join(" ") || "Test argument",
}).then(r => console.log(JSON.stringify(r, null, 2)))
  .catch(e => { console.error("Error:", e.message); process.exit(1); });
