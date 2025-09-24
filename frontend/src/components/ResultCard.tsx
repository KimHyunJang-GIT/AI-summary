import React from "react";
import type { SearchHit } from "../api/client";

type Props = {
  hit: SearchHit;
  index: number;
  onOpen: (hit: SearchHit) => void;
  onOpenFolder: (hit: SearchHit) => void;
  onPin: (hit: SearchHit) => void;
  onLike: (hit: SearchHit) => void;
  onDislike: (hit: SearchHit) => void;
};

export const ResultCard: React.FC<Props> = ({
  hit,
  index,
  onOpen,
  onOpenFolder,
  onPin,
  onLike,
  onDislike,
}) => {
  const similarity = hit.vector_similarity
    ? `${(hit.vector_similarity * 100).toFixed(1)}%`
    : "-";
  const combined = hit.combined_score?.toFixed(3) ?? "-";

  return (
    <div className="border rounded p-3 shadow-sm hover:shadow-md transition">
      <div className="flex items-center justify-between">
        <div className="font-semibold truncate" title={hit.path}>
          {index + 1}. {hit.path}
        </div>
        <span className="text-xs opacity-60">{hit.ext}</span>
      </div>
      <div className="text-xs text-gray-600 mt-1">
        유사도 {similarity} · 종합 {combined}
      </div>
      {hit.preview && (
        <div className="text-sm mt-2 line-clamp-3 text-gray-700">{hit.preview}</div>
      )}
      {hit.match_reasons && hit.match_reasons.length > 0 && (
        <ul className="text-xs mt-2 list-disc list-inside text-gray-500">
          {hit.match_reasons.slice(0, 4).map((reason, idx) => (
            <li key={idx}>{reason}</li>
          ))}
        </ul>
      )}
      <div className="flex gap-2 mt-3">
        <button className="btn" onClick={() => onOpen(hit)}>
          열기
        </button>
        <button className="btn" onClick={() => onOpenFolder(hit)}>
          폴더
        </button>
        <button className="btn" onClick={() => onPin(hit)}>
          핀
        </button>
        <button className="btn" onClick={() => onLike(hit)}>
          👍
        </button>
        <button className="btn" onClick={() => onDislike(hit)}>
          👎
        </button>
      </div>
    </div>
  );
};
