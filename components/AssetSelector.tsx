interface AssetSelectorProps {
  selectedAsset: 'XAU/USD' | 'ETH/USD'
  onAssetChange: (asset: 'XAU/USD' | 'ETH/USD') => void
}

export function AssetSelector({ selectedAsset, onAssetChange }: AssetSelectorProps) {
  const assets = ['XAU/USD', 'ETH/USD'] as const

  return (
    <div className="flex gap-3">
      {assets.map((asset) => (
        <button
          key={asset}
          onClick={() => onAssetChange(asset)}
          className={`px-6 py-2 rounded-full font-semibold text-sm transition ${
            selectedAsset === asset
              ? 'bg-gradient-to-r from-accent-primary to-accent-secondary text-white shadow-glow-purple'
              : 'bg-surface border border-border text-text-muted hover:text-text-primary hover:border-accent-primary/50'
          }`}
        >
          {asset}
        </button>
      ))}
    </div>
  )
}
